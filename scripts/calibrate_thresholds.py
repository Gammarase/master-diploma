"""
Calibrate verdict thresholds and the decisiveness margin from cached scores.

Reads a per-claim summary CSV (the notebook's ``summary.csv``) with columns
``nli_score``, ``rag_score``, ``evidence_count`` and ``gold_label``, replays
:func:`verification.aggregator.decide` over a grid of parameters, and writes
the best combination to a YAML file. No model, LLM or vector store is called.

Objective: macro-F1 over the Supported and Refuted classes, where CONFIRMED
predicts Supported, DISINFORMATION predicts Refuted, and an UNCERTAIN verdict
counts as a miss for the gold class. Abstention is the share of gold
Supported/Refuted claims that received UNCERTAIN; it must not exceed
``--max-abstention``. If no combination satisfies the constraint, the one
with the lowest abstention is reported instead.

Usage::

    python scripts/calibrate_thresholds.py \\
        --input notebooks/results/summary.csv \\
        --output configs/calibrated_thresholds.yaml \\
        --max-abstention 0.30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verification.aggregator import (  # noqa: E402
    VERDICT_CONFIRMED,
    VERDICT_DISINFORMATION,
    VERDICT_UNCERTAIN,
    DecisionConfig,
    decide,
)

__all__ = [
    "ScoredClaim",
    "CalibrationResult",
    "load_scored_claims",
    "evaluate",
    "calibrate",
    "main",
]

REQUIRED_COLUMNS = ("nli_score", "rag_score", "evidence_count", "gold_label")
GOLD_SUPPORTED = "Supported"
GOLD_REFUTED = "Refuted"
_PREDICTS = {VERDICT_CONFIRMED: GOLD_SUPPORTED, VERDICT_DISINFORMATION: GOLD_REFUTED}


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range rounded to 4 decimals."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n + 1)]


DISINFORMATION_GRID = _frange(0.20, 0.50, 0.025)
CONFIRMED_GRID = _frange(0.50, 0.80, 0.025)
MARGIN_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]


@dataclass(frozen=True)
class ScoredClaim:
    """Cached component scores and gold label for one claim."""

    nli_score: float
    rag_score: float
    evidence_count: int
    gold_label: str


@dataclass(frozen=True)
class CalibrationResult:
    """Metrics for one parameter combination."""

    threshold_disinformation: float
    threshold_confirmed: float
    decisiveness_margin: float
    macro_f1: float
    f1_supported: float
    f1_refuted: float
    abstention: float
    n_claims: int
    constraint_satisfied: bool = True


def load_scored_claims(path: str | Path) -> list[ScoredClaim]:
    """Read a summary CSV, skipping rows with missing scores.

    Raises:
        ValueError: If a required column is missing.
    """
    import pandas as pd

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Summary CSV '{path}' is missing columns: {missing}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    return [
        ScoredClaim(
            nli_score=float(r.nli_score),
            rag_score=float(r.rag_score),
            evidence_count=int(r.evidence_count),
            gold_label=str(r.gold_label).strip(),
        )
        for r in df.itertuples(index=False)
    ]


def _f1(tp: int, n_pred: int, n_gold: int) -> float:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(claims: Sequence[ScoredClaim], cfg: DecisionConfig) -> CalibrationResult:
    """Replay :func:`decide` on *claims* and compute the metrics."""
    tp = {GOLD_SUPPORTED: 0, GOLD_REFUTED: 0}
    n_pred = {GOLD_SUPPORTED: 0, GOLD_REFUTED: 0}
    n_gold = {GOLD_SUPPORTED: 0, GOLD_REFUTED: 0}
    abstained = 0

    for claim in claims:
        verdict = decide(
            claim.nli_score, claim.rag_score, claim.evidence_count, cfg
        ).verdict
        predicted = _PREDICTS.get(verdict)
        if predicted is not None:
            n_pred[predicted] += 1
        if claim.gold_label in n_gold:
            n_gold[claim.gold_label] += 1
            if verdict == VERDICT_UNCERTAIN:
                abstained += 1
            elif predicted == claim.gold_label:
                tp[predicted] += 1

    f1_s = _f1(tp[GOLD_SUPPORTED], n_pred[GOLD_SUPPORTED], n_gold[GOLD_SUPPORTED])
    f1_r = _f1(tp[GOLD_REFUTED], n_pred[GOLD_REFUTED], n_gold[GOLD_REFUTED])
    n_binary = n_gold[GOLD_SUPPORTED] + n_gold[GOLD_REFUTED]
    return CalibrationResult(
        threshold_disinformation=cfg.threshold_disinformation,
        threshold_confirmed=cfg.threshold_confirmed,
        decisiveness_margin=cfg.decisiveness_margin,
        macro_f1=(f1_s + f1_r) / 2,
        f1_supported=f1_s,
        f1_refuted=f1_r,
        abstention=abstained / n_binary if n_binary else 0.0,
        n_claims=len(claims),
    )


def calibrate(
    claims: Sequence[ScoredClaim],
    nli_weight: float,
    rag_weight: float,
    max_abstention: float = 0.30,
    disinformation_grid: Iterable[float] = DISINFORMATION_GRID,
    confirmed_grid: Iterable[float] = CONFIRMED_GRID,
    margin_grid: Iterable[float] = MARGIN_GRID,
) -> CalibrationResult:
    """Grid-search thresholds and margin (design D9).

    Returns the combination with the highest macro-F1 among those with
    abstention ≤ *max_abstention* (ties: lower abstention, then the earlier
    grid point). If none qualifies, returns the lowest-abstention combination
    (ties: higher macro-F1) with ``constraint_satisfied=False``.

    Raises:
        ValueError: If *claims* is empty.
    """
    if not claims:
        raise ValueError("No scored claims to calibrate on.")

    confirmed_values = list(confirmed_grid)
    margin_values = list(margin_grid)
    best: CalibrationResult | None = None
    fallback: CalibrationResult | None = None

    for dis in disinformation_grid:
        for conf in confirmed_values:
            if conf <= dis:
                continue
            for margin in margin_values:
                result = evaluate(
                    claims,
                    DecisionConfig(nli_weight, rag_weight, dis, conf, margin),
                )
                if result.abstention <= max_abstention + 1e-12:
                    if best is None or (result.macro_f1, -result.abstention) > (
                        best.macro_f1,
                        -best.abstention,
                    ):
                        best = result
                if fallback is None or (-result.abstention, result.macro_f1) > (
                    -fallback.abstention,
                    fallback.macro_f1,
                ):
                    fallback = result

    if best is not None:
        return best
    assert fallback is not None
    return CalibrationResult(**{**fallback.__dict__, "constraint_satisfied": False})


def write_result(
    result: CalibrationResult,
    path: str | Path,
    nli_weight: float,
    rag_weight: float,
    max_abstention: float,
    source: str,
) -> None:
    """Write the chosen values and metrics as YAML."""
    import yaml

    payload = {
        "verification": {
            "thresholds": {
                "disinformation": result.threshold_disinformation,
                "confirmed": result.threshold_confirmed,
            },
            "decisiveness_margin": result.decisiveness_margin,
        },
        "metrics": {
            "macro_f1": round(result.macro_f1, 4),
            "f1_supported": round(result.f1_supported, 4),
            "f1_refuted": round(result.f1_refuted, 4),
            "abstention": round(result.abstention, 4),
            "n_claims": result.n_claims,
            "max_abstention": max_abstention,
            "constraint_satisfied": result.constraint_satisfied,
        },
        "weights": {"nli": nli_weight, "rag": rag_weight},
        "source": source,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Generated by scripts/calibrate_thresholds.py. Copy the "
            "'verification' values into configs/config.yaml by hand.\n"
        )
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _format_report(result: CalibrationResult, max_abstention: float) -> str:
    lines = []
    if not result.constraint_satisfied:
        lines.append(
            f"No combination reaches abstention <= {max_abstention:.2f}; "
            "reporting the lowest-abstention combination."
        )
    lines += [
        f"thresholds.disinformation = {result.threshold_disinformation}",
        f"thresholds.confirmed      = {result.threshold_confirmed}",
        f"decisiveness_margin       = {result.decisiveness_margin}",
        f"macro-F1 = {result.macro_f1:.4f}  "
        f"(Supported {result.f1_supported:.4f}, Refuted {result.f1_refuted:.4f})",
        f"abstention = {result.abstention:.4f} over {result.n_claims} claims",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Summary CSV with cached scores.")
    parser.add_argument(
        "--output",
        default=str(_ROOT / "configs" / "calibrated_thresholds.yaml"),
        help="Where to write the chosen values (YAML).",
    )
    parser.add_argument("--max-abstention", type=float, default=0.30)
    parser.add_argument(
        "--nli-weight", type=float, default=None, help="Default: from config.yaml."
    )
    parser.add_argument(
        "--rag-weight", type=float, default=None, help="Default: from config.yaml."
    )
    args = parser.parse_args(argv)

    nli_weight, rag_weight = args.nli_weight, args.rag_weight
    if nli_weight is None or rag_weight is None:
        from configs import get_settings

        vcfg = get_settings().verification
        nli_weight = vcfg.nli_weight if nli_weight is None else nli_weight
        rag_weight = vcfg.rag_weight if rag_weight is None else rag_weight

    claims = load_scored_claims(args.input)
    result = calibrate(claims, nli_weight, rag_weight, args.max_abstention)
    write_result(result, args.output, nli_weight, rag_weight, args.max_abstention, args.input)
    print(_format_report(result, args.max_abstention))
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
