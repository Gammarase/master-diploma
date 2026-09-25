"""Unit tests for configs settings classes."""

from __future__ import annotations

import pytest

import configs
from configs import (
    OllamaSettings,
    RetrievalSettings,
    SearchSettings,
    Settings,
    SourcePolicySettings,
    VerificationSettings,
)


class TestDefaults:
    def test_settings_expose_new_sections(self) -> None:
        s = Settings()
        assert isinstance(s.retrieval, RetrievalSettings)
        assert set(Settings.model_fields) == {
            "ollama",
            "retrieval",
            "search",
            "source_policy",
            "preprocessing",
            "claim_extraction",
            "verification",
            "logging",
        }

    def test_ollama_generation_fields(self) -> None:
        o = OllamaSettings()
        assert isinstance(o.num_ctx, int) and o.num_ctx > 0
        assert isinstance(o.seed, int)
        assert isinstance(o.think, bool)
        assert o.evidence_mode in {"web", "evidence_only"}

    def test_retrieval_fields(self) -> None:
        r = RetrievalSettings()
        assert r.candidate_k >= 1
        assert 0.0 <= r.min_relevance <= 1.0

    def test_reranker_can_be_disabled(self) -> None:
        assert RetrievalSettings(reranker_model=None).reranker_model is None

    def test_invalid_evidence_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaSettings(evidence_mode="labels_only")

    def test_fact_checked_claims_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="web"):
            OllamaSettings(evidence_mode="fact_checked_claims")

    def test_evidence_mode_defaults_to_web(self) -> None:
        assert OllamaSettings.model_fields["evidence_mode"].default == "web"

    def test_margin_default_without_yaml(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(configs, "_YAML", {})
        assert VerificationSettings().decisiveness_margin == pytest.approx(0.15)


class TestLegacyDisagreementDelta:
    def test_legacy_kwarg_maps_to_margin(self) -> None:
        with pytest.warns(DeprecationWarning, match="disagreement_delta"):
            v = VerificationSettings(disagreement_delta=0.22)
        assert v.decisiveness_margin == pytest.approx(0.22)
        assert not hasattr(v, "disagreement_delta")

    def test_new_key_wins_over_legacy(self) -> None:
        with pytest.warns(DeprecationWarning):
            v = VerificationSettings(
                disagreement_delta=0.22, decisiveness_margin=0.1
            )
        assert v.decisiveness_margin == pytest.approx(0.1)

    def test_legacy_yaml_key_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            configs, "_YAML", {"verification": {"disagreement_delta": 0.3}}
        )
        with pytest.warns(DeprecationWarning):
            v = VerificationSettings()
        assert v.decisiveness_margin == pytest.approx(0.3)

    def test_new_yaml_key_used_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
    ) -> None:
        monkeypatch.setattr(
            configs, "_YAML", {"verification": {"decisiveness_margin": 0.2}}
        )
        v = VerificationSettings()
        assert v.decisiveness_margin == pytest.approx(0.2)
        assert not [w for w in recwarn if w.category is DeprecationWarning]


class TestWebRetrievalSettings:
    def test_defaults_load(self) -> None:
        s = Settings()
        assert isinstance(s.search, SearchSettings)
        assert isinstance(s.source_policy, SourcePolicySettings)
        r = s.retrieval
        assert r.top_k >= 1
        assert r.max_passages_per_source >= 1
        assert r.max_passage_chars > 0
        assert r.max_search_requests >= 1
        assert r.max_pages_per_claim >= 1
        assert r.cache_mode in {"read_write", "read_only", "off"}
        assert s.search.site_filter in {"grouped", "per_domain", "none"}
        assert s.search.site_group_size >= 1
        assert s.source_policy.mode in {"strict", "lenient", "off"}

    def test_invalid_policy_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            SourcePolicySettings(mode="loose")

    def test_invalid_site_filter_rejected(self) -> None:
        with pytest.raises(ValueError):
            SearchSettings(site_filter="any")

    def test_invalid_cache_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            RetrievalSettings(cache_mode="sometimes")

    def test_unknown_trust_range_checked(self) -> None:
        with pytest.raises(ValueError):
            SourcePolicySettings(unknown_trust=1.5)

    def test_search_base_url_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_BASE_URL", "http://searx.test:9999")
        assert Settings().search.base_url == "http://searx.test:9999"

    def test_source_policy_mode_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOURCE_POLICY_MODE", "off")
        assert Settings().source_policy.mode == "off"
