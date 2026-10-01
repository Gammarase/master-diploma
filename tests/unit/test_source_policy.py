"""Unit tests for retrieval.source_policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from exceptions import RetrievalError
from retrieval.source_policy import SourcePolicy, publisher_for

_DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "configs" / "source_policy.yaml"


def _policy(mode: str = "strict", **extra: object) -> SourcePolicy:
    data = {
        "version": 1,
        "tiers": {
            "fact_checkers": {
                "trust": 1.0,
                "domains": ["politifact.com", "reuters.com/fact-check"],
            },
            "wire_agencies": {"trust": 0.95, "domains": ["reuters.com", "apnews.com"]},
            "major_outlets": {"trust": 0.8, "domains": ["bbc.co.uk", "rt.com"]},
        },
        "blocklist": ["rt.com"],
    }
    data.update(extra)
    return SourcePolicy(data, mode=mode, unknown_trust=0.3)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def default_policy() -> SourcePolicy:
    return SourcePolicy.from_file(_DEFAULT_POLICY)


class TestPolicyFile:
    def test_default_tiers_present(self, default_policy: SourcePolicy) -> None:
        assert default_policy.trust_for("https://www.politifact.com/x") == 1.0
        assert default_policy.trust_for("https://apnews.com/article/x") == 0.95
        assert default_policy.trust_for("https://www.who.int/news") == 0.9
        assert default_policy.trust_for("https://www.bbc.com/news/1") == 0.8

    def test_default_policy_has_version(self, default_policy: SourcePolicy) -> None:
        assert default_policy.version

    def test_relative_path_resolved_from_project_root(self) -> None:
        policy = SourcePolicy.from_file("configs/source_policy.yaml")
        assert policy.match("https://apnews.com/x") is not None

    def test_from_settings(self) -> None:
        settings = MagicMock()
        settings.source_policy.path = str(_DEFAULT_POLICY)
        settings.source_policy.mode = "lenient"
        settings.source_policy.unknown_trust = 0.25
        policy = SourcePolicy.from_settings(settings)
        assert policy.mode == "lenient"
        assert policy.trust_for("https://random-blog.example/") == 0.25

    def test_invalid_trust_value(self) -> None:
        data = {"tiers": {"shady": {"trust": 1.5, "domains": ["x.com"]}}}
        with pytest.raises(RetrievalError, match=r"shady.*1\.5"):
            SourcePolicy(data)

    def test_non_numeric_trust(self) -> None:
        with pytest.raises(RetrievalError, match="shady"):
            SourcePolicy({"tiers": {"shady": {"trust": "high", "domains": []}}})

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(RetrievalError, match="not found"):
            SourcePolicy.from_file(tmp_path / "nope.yaml")

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("tiers: [unclosed", encoding="utf-8")
        with pytest.raises(RetrievalError, match="could not be parsed"):
            SourcePolicy.from_file(p)

    @pytest.mark.parametrize(
        "data",
        [
            {},
            {"tiers": ["a"]},
            {"tiers": {"a": "b"}},
            {"tiers": {"a": {"trust": 1.0, "domains": "x.com"}}},
            {"tiers": {}, "blocklist": "rt.com"},
        ],
    )
    def test_malformed_structure(self, data: dict) -> None:
        with pytest.raises(RetrievalError):
            SourcePolicy(data)

    def test_invalid_mode(self) -> None:
        with pytest.raises(RetrievalError, match="mode"):
            _policy(mode="loose")


class TestMatching:
    def test_subdomain_matches(self) -> None:
        policy = _policy()
        m = policy.match("https://www.bbc.co.uk/news/world-123")
        assert m is not None and m.trust == 0.8 and m.tier == "major_outlets"

    @pytest.mark.parametrize(
        "url", ["https://reuters.com.evil.io/x", "https://notreuters.com/x"]
    )
    def test_look_alike_domain_rejected(self, url: str) -> None:
        policy = _policy()
        assert policy.match(url) is None
        assert policy.allows(url) is False

    def test_path_prefix_wins(self) -> None:
        m = _policy().match("https://www.reuters.com/fact-check/abc")
        assert m is not None
        assert (m.tier, m.trust, m.entry) == ("fact_checkers", 1.0, "reuters.com/fact-check")

    def test_path_prefix_needs_segment_boundary(self) -> None:
        m = _policy().match("https://www.reuters.com/fact-checking-tips")
        assert m is not None and m.tier == "wire_agencies"

    def test_case_insensitive(self) -> None:
        m = _policy().match("HTTPS://WWW.Reuters.COM/Fact-Check/ABC")
        assert m is not None and m.tier == "fact_checkers"

    def test_port_and_trailing_dot_ignored(self) -> None:
        assert _policy().match("https://apnews.com.:443/x") is not None

    def test_empty_url(self) -> None:
        assert _policy().match("") is None


class TestModes:
    def test_strict_drops_unknown(self) -> None:
        assert _policy("strict").allows("https://random-blog.example/post") is False

    def test_strict_keeps_known(self) -> None:
        assert _policy("strict").allows("https://apnews.com/x") is True

    def test_lenient_keeps_unknown_with_low_trust(self) -> None:
        policy = _policy("lenient")
        url = "https://random-blog.example/post"
        assert policy.allows(url) is True
        assert policy.trust_for(url) == 0.3
        assert policy.tier_for(url) == ""

    @pytest.mark.parametrize("mode", ["strict", "lenient"])
    def test_blocklist_beats_tiers(self, mode: str) -> None:
        policy = _policy(mode)
        assert policy.match("https://www.rt.com/news/1") is not None
        assert policy.allows("https://www.rt.com/news/1") is False

    def test_off_allows_everything(self) -> None:
        policy = _policy("off")
        assert policy.allows("https://random-blog.example/") is True
        assert policy.allows("https://rt.com/x") is True

    def test_off_blocklisted_gets_unknown_trust(self) -> None:
        policy = _policy("off")
        assert policy.trust_for("https://rt.com/x") == 0.3
        assert policy.tier_for("https://rt.com/x") == ""
        assert policy.trust_for("https://random-blog.example/") == 0.3
        assert policy.trust_for("https://apnews.com/x") == 0.95


class TestSiteGroups:
    def _many(self) -> SourcePolicy:
        data = {
            "tiers": {
                # Declared out of trust order on purpose.
                "major": {"trust": 0.8, "domains": [f"major{i}.com" for i in range(10)]},
                "fact": {"trust": 1.0, "domains": [f"fact{i}.org" for i in range(8)]},
                "wire": {"trust": 0.95, "domains": [f"wire{i}.com" for i in range(7)]},
            }
        }
        return SourcePolicy(data)

    def test_25_domains_size_10(self) -> None:
        groups = self._many().site_groups(10)
        assert [len(g) for g in groups] == [10, 10, 5]
        assert groups[0][:8] == [f"fact{i}.org" for i in range(8)]
        assert groups[0][8:] == ["wire0.com", "wire1.com"]
        flat = [d for g in groups for d in g]
        assert len(flat) == len(set(flat)) == 25

    def test_path_entry_reduced_and_deduped(self) -> None:
        groups = _policy().site_groups(10)
        flat = [d for g in groups for d in g]
        assert flat.count("reuters.com") == 1
        assert not any("/" in d for d in flat)
        assert flat[:2] == ["politifact.com", "reuters.com"]

    def test_blocklisted_domain_skipped(self) -> None:
        flat = [d for g in _policy().site_groups(10) for d in g]
        assert "rt.com" not in flat

    def test_size_one(self) -> None:
        groups = _policy().site_groups(1)
        assert all(len(g) == 1 for g in groups)
        assert [g[0] for g in groups] == [
            "politifact.com",
            "reuters.com",
            "apnews.com",
            "bbc.co.uk",
        ]

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            _policy().site_groups(0)

    def test_default_policy_groups(self, default_policy: SourcePolicy) -> None:
        groups = default_policy.site_groups(10)
        assert groups and all(1 <= len(g) <= 10 for g in groups)
        assert groups[0][0] == "politifact.com"
        assert "meduza.io" in groups[-1]


class TestPublisher:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.bbc.co.uk/news/x", "bbc.co.uk"),
            ("https://apnews.com/article/xyz", "apnews.com"),
            ("https://news.un.org/en/story", "un.org"),
            ("http://localhost:8080/x", "localhost"),
            ("", ""),
        ],
    )
    def test_registrable_domain(self, url: str, expected: str) -> None:
        assert publisher_for(url) == expected
