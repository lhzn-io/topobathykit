/**
 * provenance-utils.js — Shared DEM provenance visualization utilities.
 *
 * Used by both topobathykit/dem-viewer.html and coastal-sim/hydro_viewer.html
 * to render per-pixel data source attribution on Three.js terrain meshes.
 *
 * Usage:
 *   <script src="provenance-utils.js"></script>
 *   // After loading DEM and provenance binary:
 *   const prov = new ProvenanceOverlay(provenanceUint32Array, provenanceDict, width, height);
 *   // In raycaster hit handler (Shift+Click):
 *   const info = prov.query(pixelIndex);
 *   // For color overlay mode:
 *   const color = prov.color(pixelIndex);
 */

const ProvenanceOverlay = (() => {
    // Perceptually distinct palette for up to 12 source tiers.
    // Index 0 = no data (transparent/dark gray).
    const PALETTE = [
        [0.15, 0.15, 0.15],  // 0: no data
        [0.50, 0.50, 0.50],  // 1: GEBCO (muted gray — global filler)
        [0.85, 0.72, 0.50],  // 2: USGS 3DEP (warm tan — land)
        [0.45, 0.75, 0.45],  // 3: NCEI CUDEM (pale green — regional coastal)
        [0.30, 0.55, 0.90],  // 4: NOAA BlueTopo (ocean blue — compiled bathy)
        [0.00, 0.85, 0.85],  // 5: Topobathy Lidar (bright cyan — intertidal)
        [0.90, 0.30, 0.70],  // 6: NCEI BAG surveys (vivid magenta — truth)
        [1.00, 0.85, 0.20],  // 7: (yellow — spare)
        [0.70, 0.30, 0.90],  // 8: (purple — spare)
        [1.00, 0.50, 0.20],  // 9: (orange — spare)
        [0.20, 0.90, 0.50],  // 10: (mint — spare)
        [0.90, 0.20, 0.30],  // 11: (red — spare)
        [0.40, 0.70, 0.80],  // 12: (slate blue — spare)
    ];

    class ProvenanceOverlay {
        /**
         * @param {Uint32Array} buffer - Per-pixel source IDs (row-major, north-first)
         * @param {Object} legend - Maps source ID (string) → {provider, product, ...}
         * @param {number} width - Grid width in pixels
         * @param {number} height - Grid height in pixels
         */
        constructor(buffer, legend, width, height) {
            this.buffer = buffer;
            this.legend = legend || {};
            this.width = width;
            this.height = height;
            this._buildStats();
        }

        /** Get source info for a pixel index. */
        query(index) {
            if (!this.buffer || index < 0 || index >= this.buffer.length) {
                return { id: 0, provider: 'Unknown', product: '', color: PALETTE[0] };
            }
            const id = this.buffer[index];
            if (id === 0) {
                return { id: 0, provider: 'No data', product: '', color: PALETTE[0] };
            }
            const entry = this.legend[String(id)] || this.legend[id];
            const name = entry ? (entry.name || entry.provider || `Source ${id}`) : `Source ${id}`;
            const provider = entry ? (entry.provider || '') : '';
            const resolution = (entry && entry.resolution && entry.resolution !== 'Unknown') ? entry.resolution : '';
            const color = PALETTE[id % PALETTE.length] || PALETTE[0];
            return { id, name, provider, resolution, color };
        }

        /** Get RGB color [r, g, b] (0-1) for a pixel index. */
        color(index) {
            if (!this.buffer || index < 0 || index >= this.buffer.length) return PALETTE[0];
            const id = this.buffer[index];
            return PALETTE[id % PALETTE.length] || PALETTE[0];
        }

        /** Get THREE.Color for a pixel index. */
        threeColor(index) {
            const [r, g, b] = this.color(index);
            return { r, g, b };
        }

        /** Apply provenance colors to a mesh's vertex color attribute.
         *  vertexToPixel maps vertex index → DEM pixel index. */
        applyToMesh(geometry, vertexToPixel) {
            const colors = geometry.attributes.color;
            if (!colors) return;
            for (let vi = 0; vi < colors.count; vi++) {
                const pi = vertexToPixel(vi);
                const [r, g, b] = this.color(pi);
                colors.setXYZ(vi, r, g, b);
            }
            colors.needsUpdate = true;
        }

        /** Build pixel count stats per source. */
        _buildStats() {
            this.stats = {};
            if (!this.buffer) return;
            for (let i = 0; i < this.buffer.length; i++) {
                const id = this.buffer[i];
                this.stats[id] = (this.stats[id] || 0) + 1;
            }
        }

        /** Get sorted stats array [{id, provider, count, pct}], deduplicated by name. */
        getStats() {
            const total = this.width * this.height;
            const byName = new Map();
            for (const [id, count] of Object.entries(this.stats)) {
                const intId = parseInt(id);
                if (intId === 0) continue;
                // Look up legend directly by source ID — query() takes a pixel index,
                // not a source ID, so calling query(intId) here was reading a random pixel.
                const entry = this.legend[String(intId)] || this.legend[intId];
                const name = entry ? (entry.name || entry.provider || `Source ${intId}`) : `Source ${intId}`;
                const provider = entry ? (entry.provider || '') : '';
                const resolution = (entry && entry.resolution && entry.resolution !== 'Unknown') ? entry.resolution : '';
                const color = PALETTE[intId % PALETTE.length] || PALETTE[0];
                const key = name;
                if (byName.has(key)) {
                    byName.get(key).count += count;
                } else {
                    byName.set(key, { id: intId, name, provider, resolution, count, color });
                }
            }
            return Array.from(byName.values())
                .map(s => ({ ...s, pct: ((s.count / total) * 100).toFixed(1) }))
                .sort((a, b) => b.count - a.count);
        }

        /** Generate HTML for a provenance stats panel. */
        statsHTML() {
            const stats = this.getStats();
            if (stats.length === 0) return '<em>No provenance data</em>';
            let html = '<div style="font-size:11px;">';
            html += '<div style="color:var(--accent,#4af); font-weight:600; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">Data Sources</div>';
            for (const s of stats) {
                const [r, g, b] = s.color;
                const css = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
                const label = s.resolution ? `${s.name} (${s.resolution})` : s.name;
                html += `<div style="display:flex; align-items:center; gap:5px; margin:2px 0;" title="${s.name}">`;
                html += `<span style="display:inline-block; width:10px; height:10px; background:${css}; border-radius:2px; flex-shrink:0;"></span>`;
                html += `<span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${label}</span>`;
                html += `<span style="color:var(--muted,#888); flex-shrink:0;">${s.pct}%</span>`;
                html += `</div>`;
            }
            html += '</div>';
            return html;
        }

        /** Generate tooltip HTML for a pixel. */
        tooltipHTML(index) {
            const info = this.query(index);
            if (info.id === 0) return '';
            const [r, g, b] = info.color;
            const css = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
            let html = `<span style="color:${css}; font-weight:600;">${info.name}</span>`;
            if (info.provider) {
                html += `<br><span style="color:#ccc; font-size:10px;">${info.provider}`;
                if (info.resolution) html += ` · ${info.resolution}`;
                html += `</span>`;
            }
            return html;
        }

        /** The color palette (for legend rendering). */
        static get PALETTE() { return PALETTE; }
    }

    return ProvenanceOverlay;
})();

// Export for module contexts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProvenanceOverlay;
}
