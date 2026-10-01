"""Unit tests for retrieval.cache.WebCache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from retrieval.cache import CACHE_FILENAME, WebCache, utc_now_iso

_RESP = {"query": "q site:a.com", "hits": [{"url": "https://a.com/x"}], "engine_errors": []}


def _clock(value: str):
    return lambda: value


class TestRoundTrip:
    def test_search_round_trip(self, tmp_path: Path) -> None:
        cache = WebCache(tmp_path)
        assert cache.get_search("searxng", "q site:a.com", "en") is None
        cache.put_search("searxng", "q site:a.com", "en", _RESP)
        assert cache.get_search("searxng", "q site:a.com", "en") == _RESP
        # Key includes language and backend.
        assert cache.get_search("searxng", "q site:a.com", "uk") is None
        assert cache.get_search("other", "q site:a.com", "en") is None
        assert (tmp_path / CACHE_FILENAME).is_file()

    def test_page_round_trip(self, tmp_path: Path) -> None:
        cache = WebCache(tmp_path, now=_clock("2026-09-25T10:00:00Z"))
        content = {"text": "Body", "title": "T", "date": "2022-08-12"}
        cache.put_page("https://a.com/x", "https://www.a.com/x", "ok", content)
        entry = cache.get_page("https://a.com/x")
        assert entry == {
            "final_url": "https://www.a.com/x",
            "status": "ok",
            "content": content,
            "retrieved_at": "2026-09-25T10:00:00Z",
        }

    def test_negative_page_result(self, tmp_path: Path) -> None:
        cache = WebCache(tmp_path)
        cache.put_page("https://a.com/x", "https://a.com/x", "403", None)
        entry = cache.get_page("https://a.com/x")
        assert entry is not None
        assert entry["status"] == "403" and entry["content"] is None

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        WebCache(tmp_path).put_search("searxng", "q", "en", _RESP)
        assert WebCache(tmp_path).get_search("searxng", "q", "en") == _RESP

    def test_creates_directory(self, tmp_path: Path) -> None:
        WebCache(tmp_path / "nested" / "cache")
        assert (tmp_path / "nested" / "cache" / CACHE_FILENAME).is_file()


class TestModes:
    def test_read_only_never_writes(self, tmp_path: Path) -> None:
        WebCache(tmp_path).put_search("searxng", "q", "en", _RESP)
        ro = WebCache(tmp_path, mode="read_only")
        assert ro.offline is True
        ro.put_search("searxng", "new", "en", _RESP)
        ro.put_page("https://a.com", "https://a.com", "ok", {"text": "x"})
        assert ro.get_search("searxng", "new", "en") is None
        assert ro.get_page("https://a.com") is None
        assert ro.get_search("searxng", "q", "en") == _RESP
        ro.close()
        assert WebCache(tmp_path).get_search("searxng", "new", "en") is None

    def test_read_only_missing_file(self, tmp_path: Path) -> None:
        ro = WebCache(tmp_path / "absent", mode="read_only")
        assert ro.get_search("searxng", "q", "en") is None
        ro.put_search("searxng", "q", "en", _RESP)
        assert not (tmp_path / "absent").exists()

    def test_off_never_reads(self, tmp_path: Path) -> None:
        WebCache(tmp_path).put_search("searxng", "q", "en", _RESP)
        off = WebCache(tmp_path, mode="off")
        assert off.offline is False
        assert off.get_search("searxng", "q", "en") is None
        off.put_page("https://a.com", "https://a.com", "ok", {"text": "x"})
        assert WebCache(tmp_path).get_page("https://a.com") is None

    def test_invalid_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WebCache(tmp_path, mode="sometimes")  # type: ignore[arg-type]


class TestExpiry:
    def test_expired_entry_ignored(self, tmp_path: Path) -> None:
        writer = WebCache(tmp_path, now=_clock("2026-01-01T00:00:00Z"))
        writer.put_search("searxng", "q", "en", _RESP)
        writer.put_page("https://a.com", "https://a.com", "ok", {"text": "x"})
        later = WebCache(
            tmp_path, max_age_days=7, now=_clock("2026-01-10T00:00:00Z")
        )
        assert later.get_search("searxng", "q", "en") is None
        assert later.get_page("https://a.com") is None

    def test_fresh_entry_used(self, tmp_path: Path) -> None:
        writer = WebCache(tmp_path, now=_clock("2026-01-01T00:00:00Z"))
        writer.put_search("searxng", "q", "en", _RESP)
        later = WebCache(
            tmp_path, max_age_days=7, now=_clock("2026-01-05T00:00:00Z")
        )
        assert later.get_search("searxng", "q", "en") == _RESP

    def test_no_max_age_never_expires(self, tmp_path: Path) -> None:
        WebCache(tmp_path, now=_clock("2000-01-01T00:00:00Z")).put_search(
            "searxng", "q", "en", _RESP
        )
        assert WebCache(tmp_path).get_search("searxng", "q", "en") == _RESP


def test_failed_search_not_stored(tmp_path: Path) -> None:
    cache = WebCache(tmp_path)
    cache.put_search("searxng", "q", "en", {"hits": []}, failed=True)
    assert cache.get_search("searxng", "q", "en") is None


def test_from_settings(tmp_path: Path) -> None:
    settings = MagicMock()
    settings.retrieval.cache_dir = str(tmp_path)
    settings.retrieval.cache_mode = "off"
    settings.retrieval.cache_max_age_days = None
    assert WebCache.from_settings(settings).mode == "off"


def test_utc_now_iso_format() -> None:
    value = utc_now_iso()
    assert value.endswith("Z") and len(value) == 20
