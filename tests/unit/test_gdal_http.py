import logging

import pytest

from topobathykit import gdal_http


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("rasterio._env", logging.WARNING, __file__, 1, message, (), None)


def test_defaults_fill_unset_variables_and_keep_set_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in gdal_http.GDAL_HTTP_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "7")

    gdal_http.apply_gdal_http_defaults()

    import os

    assert os.environ["GDAL_HTTP_TIMEOUT"] == "7"
    for key, value in gdal_http.GDAL_HTTP_DEFAULTS.items():
        if key != "GDAL_HTTP_TIMEOUT":
            assert os.environ[key] == value


def test_package_import_applies_defaults() -> None:
    import os

    import topobathykit  # noqa: F401

    for key in gdal_http.GDAL_HTTP_DEFAULTS:
        assert key in os.environ


def test_curl_failures_are_rate_limited_and_other_messages_pass() -> None:
    f = gdal_http.RepeatedCurlFailureFilter(limit=3)
    curl = [_record("CPLE_AppDefined in curl_multi_poll() failed") for _ in range(6)]
    passed = [f.filter(r) for r in curl]

    assert passed == [True, True, True, True, False, False]
    assert "suppressed" in curl[3].getMessage()
    assert f.filter(_record("Request for 0-16383 failed with response_code=0")) is False
    assert f.filter(_record("CPLE_AppDefined in some other GDAL warning")) is True


def test_curl_filter_is_installed_once_on_rasterio_logger() -> None:
    gdal_http.install_curl_failure_filter()
    gdal_http.install_curl_failure_filter()

    filters = logging.getLogger("rasterio._env").filters
    assert sum(isinstance(f, gdal_http.RepeatedCurlFailureFilter) for f in filters) == 1
