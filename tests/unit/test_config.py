"""Unit tests for configs settings classes."""

from __future__ import annotations

import pytest

import configs
from configs import (
    IndexingSettings,
    OllamaSettings,
    RetrievalSettings,
    Settings,
    VerificationSettings,
)


class TestDefaults:
    def test_settings_expose_new_sections(self) -> None:
        s = Settings()
        assert isinstance(s.retrieval, RetrievalSettings)
        assert isinstance(s.indexing, IndexingSettings)

    def test_ollama_generation_fields(self) -> None:
        o = OllamaSettings()
        assert isinstance(o.num_ctx, int) and o.num_ctx > 0
        assert isinstance(o.seed, int)
        assert isinstance(o.think, bool)
        assert o.evidence_mode in {"evidence_only", "fact_checked_claims"}

    def test_retrieval_fields(self) -> None:
        r = RetrievalSettings()
        assert r.candidate_k >= 1
        assert 0.0 <= r.min_relevance <= 1.0
        assert r.max_passages_per_record >= 1

    def test_reranker_can_be_disabled(self) -> None:
        assert RetrievalSettings(reranker_model=None).reranker_model is None

    def test_indexing_fields(self) -> None:
        assert IndexingSettings().max_passage_chars > 0

    def test_invalid_evidence_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaSettings(evidence_mode="labels_only")

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
