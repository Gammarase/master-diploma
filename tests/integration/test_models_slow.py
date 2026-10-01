"""
Smoke tests against the real Hugging Face models configured in config.yaml.

These download and load models, so they are skipped unless pytest is run
with ``--run-slow``.
"""

from __future__ import annotations

import pytest

from configs import get_settings

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def nli_verifier():
    from verification.nli_verifier import NLIVerifier

    verifier = NLIVerifier(get_settings())
    verifier._load_model()
    return verifier


class TestMultilingualNLI:
    """Scenarios from specs/nli-verification/spec.md."""

    def test_entailing_evidence(self, nli_verifier) -> None:
        result = nli_verifier.verify_single(
            claim="Ukraine to get long-range missiles in latest US aid",
            evidence="Ukraine received long-range missiles in the latest US aid package.",
        )
        assert result.predicted_label == "entailment"
        assert result.entailment_score > max(
            result.neutral_score, result.contradiction_score
        )
        assert result.support > 0

    def test_russian_contradiction(self, nli_verifier) -> None:
        result = nli_verifier.verify_single(
            claim="5 июня в Москве прогремели взрывы.",
            evidence=(
                "Мэр Москвы опроверг сообщения о взрывах: 5 июня никаких "
                "взрывов в городе не было."
            ),
        )
        assert result.contradiction_score > result.entailment_score

    def test_probabilities_sum_to_one(self, nli_verifier) -> None:
        from retrieval.evidence import RetrievedEvidence

        evidences = [
            RetrievedEvidence(evidence_id=f"v{i}", score=0.9, evidence_text=text)
            for i, text in enumerate(
                [
                    "Україна отримала далекобійні ракети.",
                    "Погода в Києві була сонячною.",
                    "США не надавали Україні ракет.",
                    "乌克兰获得了远程导弹。",
                ]
            )
        ]
        results = nli_verifier.verify_batch(
            "Ukraine received long-range missiles.", evidences
        )
        assert len(results) == 4
        for r, ev in zip(results, evidences):
            assert r.evidence_text == ev.evidence_text
            total = r.entailment_score + r.neutral_score + r.contradiction_score
            assert total == pytest.approx(1.0, abs=0.01)


@pytest.fixture(scope="module")
def claim_pipeline():
    from claim_extraction.extractor import ClaimExtractor
    from claim_extraction.ner_module import NERModule
    from preprocessing import PreprocessingPipeline

    settings = get_settings()
    preprocessing = PreprocessingPipeline(settings)
    extractor = ClaimExtractor(
        NERModule(preprocessing.tokenizer), preprocessing.tokenizer, settings
    )
    return preprocessing, extractor, settings


class TestMultilingualClaimExtraction:
    """Scenarios from specs/claim-extraction/spec.md."""

    @pytest.mark.parametrize(
        "text,lang",
        [
            ("У Києві 5 червня пролунали вибухи, повідомив мер міста.", "uk"),
            ("Минобороны России заявило об уничтожении 20 украинских танков.", "ru"),
        ],
    )
    def test_factual_sentence_extracted(self, claim_pipeline, text, lang) -> None:
        preprocessing, extractor, settings = claim_pipeline
        preprocessed = preprocessing.process(text)
        assert preprocessed.language == lang
        claims = extractor.extract_claims(preprocessed)
        assert len(claims) == 1
        assert (
            claims[0].checkworthy_score
            >= settings.claim_extraction.checkworthy_threshold
        )

    def test_chinese_opinion_not_extracted(self, claim_pipeline) -> None:
        preprocessing, extractor, _ = claim_pipeline
        preprocessed = preprocessing.process("我觉得这部电影很好看。")
        assert preprocessed.language == "zh"
        assert extractor.extract_claims(preprocessed) == []
