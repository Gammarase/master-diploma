"""Unit tests for scripts/calibrate_thresholds.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.calibrate_thresholds import (
    CONFIRMED_GRID,
    DISINFORMATION_GRID,
    MARGIN_GRID,
    ScoredClaim,
    calibrate,
    evaluate,
    load_scored_claims,
    main,
)
from verification.aggregator import DecisionConfig


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _separable_rows() -> list[dict]:
    """NLI neutral; RAG 0.70 for Supported and 0.30 for Refuted claims.

    With margin ≤ 0.15 the RAG score is decisive and becomes the final score,
    so any disinformation threshold > 0.30 and confirmed threshold ≤ 0.70
    classifies everything correctly. With margin ≥ 0.20 RAG is not decisive
    and final scores shrink towards 0.5 (0.64 / 0.36).
    """
    rows = []
    for i in range(10):
        rows.append({"nli_score": 0.5, "rag_score": 0.70, "evidence_count": 5, "gold_label": "Supported"})
        rows.append({"nli_score": 0.5, "rag_score": 0.30, "evidence_count": 5, "gold_label": "Refuted"})
    return rows


class TestGrid:
    def test_grid_matches_design(self) -> None:
        assert DISINFORMATION_GRID[0] == 0.20 and DISINFORMATION_GRID[-1] == 0.50
        assert CONFIRMED_GRID[0] == 0.50 and CONFIRMED_GRID[-1] == 0.80
        assert len(DISINFORMATION_GRID) == len(CONFIRMED_GRID) == 13
        assert MARGIN_GRID == [0.05, 0.10, 0.15, 0.20, 0.25]


class TestLoad:
    def test_skips_rows_with_missing_scores(self, tmp_path: Path) -> None:
        rows = _separable_rows()[:2] + [
            {"nli_score": None, "rag_score": None, "evidence_count": None, "gold_label": "Supported"}
        ]
        claims = load_scored_claims(_write_csv(tmp_path / "s.csv", rows))
        assert len(claims) == 2
        assert claims[0] == ScoredClaim(0.5, 0.70, 5, "Supported")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path / "s.csv", [{"nli_score": 0.5, "rag_score": 0.5}])
        with pytest.raises(ValueError, match="missing columns"):
            load_scored_claims(path)


class TestEvaluate:
    def test_abstention_counts_as_miss(self) -> None:
        claims = [
            ScoredClaim(0.5, 0.9, 3, "Supported"),
            ScoredClaim(0.5, 0.5, 3, "Supported"),  # UNCERTAIN
            ScoredClaim(0.5, 0.1, 3, "Refuted"),
            ScoredClaim(0.5, 0.9, 3, "NEI"),        # false positive for Supported
        ]
        r = evaluate(claims, DecisionConfig(0.3, 0.7, 0.35, 0.65, 0.15))
        # Supported: tp=1, predicted=2, gold=2 → P=0.5 R=0.5 F1=0.5; Refuted F1=1.
        assert r.f1_supported == pytest.approx(0.5)
        assert r.f1_refuted == pytest.approx(1.0)
        assert r.macro_f1 == pytest.approx(0.75)
        assert r.abstention == pytest.approx(1 / 3)
        assert r.n_claims == 4


class TestCalibrate:
    def test_finds_known_optimum(self, tmp_path: Path) -> None:
        claims = load_scored_claims(_write_csv(tmp_path / "s.csv", _separable_rows()))
        r = calibrate(claims, nli_weight=0.3, rag_weight=0.7, max_abstention=0.3)
        assert r.macro_f1 == pytest.approx(1.0)
        assert r.abstention == 0.0
        assert r.constraint_satisfied
        assert r.threshold_disinformation > 0.30
        assert r.threshold_confirmed <= 0.70
        # First perfect grid point in search order.
        assert (r.threshold_disinformation, r.threshold_confirmed, r.decisiveness_margin) == (
            0.325, 0.5, 0.05
        )

    def test_constraint_limits_choice(self) -> None:
        # One Supported claim sits in the middle; it can be classified only
        # with a lower confirmed threshold, which then also mislabels NEI.
        claims = [ScoredClaim(0.5, 0.9, 3, "Supported")] * 4 + [
            ScoredClaim(0.5, 0.1, 3, "Refuted")
        ] * 4 + [ScoredClaim(0.5, 0.6, 3, "Supported")] * 2
        strict = calibrate(claims, 0.3, 0.7, max_abstention=0.0)
        assert strict.abstention == 0.0
        assert strict.threshold_confirmed <= 0.3 * 0.5 + 0.7 * 0.6

    def test_unsatisfiable_constraint_falls_back(self) -> None:
        claims = [ScoredClaim(0.5, 0.9, 0, "Supported")] * 5 + [
            ScoredClaim(0.5, 0.1, 3, "Refuted")
        ] * 5
        r = calibrate(claims, 0.3, 0.7, max_abstention=0.3)
        assert r.constraint_satisfied is False
        assert r.abstention == pytest.approx(0.5)  # lowest achievable
        assert r.f1_refuted == pytest.approx(1.0)

    def test_empty_claims_raise(self) -> None:
        with pytest.raises(ValueError):
            calibrate([], 0.3, 0.7)


class TestMain:
    def test_writes_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        csv = _write_csv(tmp_path / "summary.csv", _separable_rows())
        out = tmp_path / "calibrated.yaml"
        code = main([
            "--input", str(csv), "--output", str(out),
            "--max-abstention", "0.3", "--nli-weight", "0.3", "--rag-weight", "0.7",
        ])
        assert code == 0
        data = yaml.safe_load(out.read_text(encoding="utf-8"))
        v = data["verification"]
        assert set(v["thresholds"]) == {"disinformation", "confirmed"}
        assert "decisiveness_margin" in v
        m = data["metrics"]
        for key in ("macro_f1", "f1_supported", "f1_refuted", "abstention"):
            assert key in m
        assert m["abstention"] <= 0.30
        assert m["constraint_satisfied"] is True
        assert "macro-F1 = 1.0000" in capsys.readouterr().out

    def test_reports_unsatisfiable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        rows = [{"nli_score": 0.5, "rag_score": 0.9, "evidence_count": 0, "gold_label": "Supported"}] * 3
        csv = _write_csv(tmp_path / "summary.csv", rows)
        out = tmp_path / "calibrated.yaml"
        main(["--input", str(csv), "--output", str(out)])
        assert "No combination reaches abstention" in capsys.readouterr().out
        assert yaml.safe_load(out.read_text(encoding="utf-8"))["metrics"]["constraint_satisfied"] is False
