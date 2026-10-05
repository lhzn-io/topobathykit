import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BathyServe with Parallel Workers")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9595, help="Bind port")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--debug", type=int, default=0, help="Debug level (0=INFO, 1=DEBUG)")
    parser.add_argument(
        "--limit-max-requests",
        type=int,
        default=200,
        help="Recycle worker after N requests to reclaim leaked memory (default: 200, 0=disable)",
    )

    args = parser.parse_args()

    # Set Debug Level Environment Variable
    import os

    os.environ["TOPOBATHYKIT_DEBUG"] = str(args.debug)

    # Default to 2 workers. Hydration job state is persisted to disk (JSON files),
    # so multi-worker is safe. Only the _HYDRATE_PROCESSES dict (subprocess handles)
    # is per-worker, which only affects cleanup of zombie processes.
    # Tile fusion concurrency is controlled by _FUSION_SEMAPHORE per worker.
    if args.workers is None:
        args.workers = 2
        print(f"Auto-configured workers: {args.workers}")

    # If reload is True, workers must be 1 usually, or simple reload logic.
    # Uvicorn handles reload with workers=1?? No, reload excludes workers argument.
    log_level = "debug" if args.debug >= 1 else "info"

    if args.reload:
        print("Reload enabled: Forcing workers=1")
        uvicorn.run(
            "topobathyserve.main:app", host=args.host, port=args.port, reload=True, log_level=log_level
        )
    else:
        limit_max = args.limit_max_requests if args.limit_max_requests > 0 else None
        print(
            f"Starting BathyServe on {args.host}:{args.port} with {args.workers} workers. "
            f"Log Level: {log_level}, Worker Recycle After: {limit_max or 'disabled'} requests"
        )
        uvicorn.run(
            "topobathyserve.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=log_level,
            limit_max_requests=limit_max,
        )


if __name__ == "__main__":
    main()
