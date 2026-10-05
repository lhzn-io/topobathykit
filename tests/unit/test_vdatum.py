from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from topobathykit.vdatum import VDatumNoDataError, VDatumResolver


@pytest.fixture(autouse=True)
def isolated_vdatum_cache(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the L2 SQLite cache at an empty per-test file.

    The resolver checks the on-disk cache before making the HTTP request these
    tests mock. Without isolation, the first (valid) test writes (43.0, -70.0)
    to the real cache and the error-path tests then hit that row instead of the
    mock, so the expected exceptions are never raised. The row also persists on
    disk across runs, which made the failure machine-dependent.
    """
    monkeypatch.setattr("topobathykit.vdatum.VDATUM_DB_PATH", tmp_path / "vdatum.sqlite")


def test_vdatum_valid_response() -> None:
    """Test that a valid t_z response is returned correctly."""
    with patch("topobathykit.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": "1.5"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        val = VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)
        assert val == 1.5


def test_vdatum_nodata_response_raises_value_error() -> None:
    """Test that extreme NoData values (-999999.0) raise a VDatumNoDataError."""
    with patch("topobathykit.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"t_z": "-999999.0"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(VDatumNoDataError, match="VDatum API returned NoData value"):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)


def test_vdatum_missing_tz_raises_value_error() -> None:
    """Test that a missing t_z field raises a ValueError."""
    with patch("topobathykit.vdatum.requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Out of bounds"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Clear the lru_cache for testing
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(ValueError, match="VDatum API returned no elevation data"):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)


def _ok(value: str = "1.5") -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"t_z": value}
    response.raise_for_status.return_value = None
    return response


def test_vdatum_retries_read_timeouts_with_backoff() -> None:
    """Transient timeouts are retried with exponential backoff, then succeed."""
    import requests

    from topobathykit.vdatum import VDATUM_BACKOFF_S

    with (
        patch("topobathykit.vdatum.requests.Session.get") as mock_get,
        patch("topobathykit.vdatum.time.sleep") as mock_sleep,
    ):
        mock_get.side_effect = [requests.ReadTimeout("read timed out"), requests.ReadTimeout("again"), _ok()]
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        assert VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0) == 1.5

    assert mock_get.call_count == 3
    assert [c.args[0] for c in mock_sleep.call_args_list] == [VDATUM_BACKOFF_S, 2 * VDATUM_BACKOFF_S]


def test_vdatum_gives_up_after_bounded_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistent outage raises VDatumUnavailableError after a bounded number of attempts."""
    import requests

    from topobathykit.vdatum import VDatumUnavailableError

    monkeypatch.setenv("TOPOBATHY_VDATUM_ATTEMPTS", "4")
    with (
        patch("topobathykit.vdatum.requests.Session.get") as mock_get,
        patch("topobathykit.vdatum.time.sleep"),
    ):
        mock_get.side_effect = requests.ReadTimeout("read timed out")
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(VDatumUnavailableError, match="after 4 attempt"):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)

    assert mock_get.call_count == 4


def test_vdatum_does_not_retry_client_errors() -> None:
    """A 4xx response is not transient and is raised on the first attempt."""
    import requests

    response = MagicMock()
    response.status_code = 400
    response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request", response=response)
    with (
        patch("topobathykit.vdatum.requests.Session.get", return_value=response) as mock_get,
        patch("topobathykit.vdatum.time.sleep") as mock_sleep,
    ):
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(requests.HTTPError):
            VDatumResolver.get_mllw_to_navd88_offset(43.0, -70.0)

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


def test_vdatum_outage_is_not_mistaken_for_no_coverage() -> None:
    """The robust search must not treat an outage as a NoData point and probe a ring of points."""
    import requests

    from topobathykit.vdatum import VDatumUnavailableError

    with (
        patch("topobathykit.vdatum.requests.Session.get") as mock_get,
        patch("topobathykit.vdatum.time.sleep"),
    ):
        mock_get.side_effect = requests.ConnectionError("connection refused")
        VDatumResolver.get_mllw_to_navd88_offset.cache_clear()

        with pytest.raises(VDatumUnavailableError):
            VDatumResolver.get_robust_mllw_to_navd88_offset(43.0, -70.0)

    assert mock_get.call_count == 3
