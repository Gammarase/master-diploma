"""Request and response models of the check API.

Field descriptions and examples feed the generated OpenAPI document.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

__all__ = [
    "SubmitRequest",
    "SubmitResponse",
    "CheckStatusResponse",
    "ClaimResult",
    "Citation",
    "EvidenceExcerpt",
    "ErrorResponse",
    "HealthResponse",
]

CheckStatus = Literal["queued", "running", "completed", "failed"]

_EXAMPLE_ID = "c874c8b4-f43b-47aa-bb15-b4bed539b6f8"


class SubmitRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "The Kakhovka dam was destroyed on 6 June 2023."}]
        }
    )

    text: StrictStr = Field(
        description=(
            "News text to check, in English, Ukrainian, Russian or Chinese. "
            "Leading and trailing whitespace is removed; the rest must be "
            "non-empty and at most `API_MAX_TEXT_CHARS` characters (default 20000)."
        ),
    )


class SubmitResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _EXAMPLE_ID,
                    "status": "queued",
                    "created_at": "2026-09-28T01:36:47.909900+00:00",
                }
            ]
        }
    )

    id: str = Field(description="Random UUID of the new check. Use it to fetch the result.")
    status: CheckStatus = Field(description="Always `queued` for a new check.")
    created_at: str = Field(description="Submission time, ISO 8601 UTC.")


class Citation(BaseModel):
    """Source of one evidence passage."""

    model_config = ConfigDict(extra="allow")

    passage_index: int | None = Field(None, description="Index of the passage this source backs.")
    url: str | None = Field(None, description="Address of the source page.")
    title: str | None = Field(None, description="Page title.")
    publisher: str | None = Field(None, description="Publisher domain, e.g. `apnews.com`.")
    published_at: str | None = Field(None, description="Publication date, when known.")
    retrieved_at: str | None = Field(None, description="Time the page was fetched, ISO 8601 UTC.")
    tier: str | None = Field(
        None, description="Source-policy tier, e.g. `fact_checkers`, `wire_agencies`."
    )
    trust: float | None = Field(None, description="Trust weight of the source, 0-1.")
    language: str | None = Field(None, description="Language of the page.")


class EvidenceExcerpt(BaseModel):
    """One evidence passage and how it relates to the claim."""

    model_config = ConfigDict(extra="allow")

    passage_index: int | None = Field(None, description="Index of the passage.")
    text: str | None = Field(None, description="Passage text (shortened).")
    relevance_score: float | None = Field(None, description="Reranker relevance, 0-1.")
    stance: str | None = Field(
        None, description="NLI stance toward the claim: `supports`, `contradicts` or `neutral`."
    )


class ClaimResult(BaseModel):
    """Verification result for one check-worthy claim found in the text."""

    model_config = ConfigDict(extra="allow")

    claim_text: str = Field(description="The claim extracted from the submitted text.")
    verdict: str = Field(
        description=(
            "`CONFIRMED` (supported by evidence), `DISINFORMATION` (contradicted "
            "by evidence) or `UNCERTAIN` (not enough evidence, or the methods disagree)."
        ),
        examples=["CONFIRMED"],
    )
    final_score: float = Field(
        description="Truthfulness score from 0 (false) to 1 (true); 0.5 means undecided."
    )
    explanation: str = Field(description="Short human-readable explanation of the verdict.")
    component_scores: dict[str, float] = Field(
        description="Scores of the individual methods: `nli` (entailment model) and `rag` (LLM)."
    )
    evidence_excerpts: list[EvidenceExcerpt] = Field(
        description="Evidence passages used for the verdict."
    )
    citations: list[Citation] = Field(description="Sources of the evidence passages.")
    language: str = Field(description="Detected language of the claim: `en`, `uk`, `ru` or `zh`.")
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Diagnostic details: models used (`models`), search queries "
            "(`search_queries`), search status (`search_status`: `ok`, "
            "`no_results` or `unavailable`), weights and the reason for an "
            "`UNCERTAIN` verdict (`uncertainty_reason`). Keys may change between versions."
        ),
    )


class CheckStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": _EXAMPLE_ID,
                    "status": "completed",
                    "text": "The Kakhovka dam was destroyed in 2023.",
                    "created_at": "2026-09-28T01:36:47.909900+00:00",
                    "started_at": "2026-09-28T01:36:47.912753+00:00",
                    "finished_at": "2026-09-28T01:39:41.079260+00:00",
                    "queue_position": None,
                    "results": [
                        {
                            "claim_text": "The Kakhovka dam was destroyed in 2023.",
                            "verdict": "CONFIRMED",
                            "final_score": 0.95,
                            "explanation": (
                                "The claim is supported by 5 evidence passage(s) "
                                "with an overall confidence score of 0.95."
                            ),
                            "component_scores": {"nli": 0.2112, "rag": 0.95},
                            "evidence_excerpts": [
                                {
                                    "passage_index": 0,
                                    "text": "Collapse of Kakhovka dam in Ukraine triggers emergency ...",
                                    "relevance_score": 0.9829,
                                    "stance": "supports",
                                }
                            ],
                            "citations": [
                                {
                                    "passage_index": 0,
                                    "url": "https://apnews.com/article/russia-ukraine-war-kakhovka-dam-flood-evacuation-eecc9952c2d9f500c38b0a873f69438c",
                                    "publisher": "apnews.com",
                                    "title": "Collapse of Kakhovka dam in Ukraine triggers emergency | AP News",
                                    "published_at": "2023-06-12",
                                    "retrieved_at": "2026-09-28T01:38:44Z",
                                    "tier": "wire_agencies",
                                    "trust": 0.95,
                                    "language": "en",
                                }
                            ],
                            "language": "en",
                            "processing_metadata": {
                                "search_status": "ok",
                                "search_queries": [
                                    "The Kakhovka dam was destroyed in 2023.",
                                    "Kakhovka dam destruction 2023 news reports fact-checks",
                                ],
                                "models": {"llm": "qwen3:14b"},
                            },
                        }
                    ],
                    "error": None,
                }
            ]
        }
    )

    id: str = Field(description="UUID of the check.")
    status: CheckStatus = Field(
        description=(
            "`queued` (waiting), `running` (being analysed), `completed` "
            "(results ready) or `failed` (see `error`)."
        )
    )
    text: str = Field(description="Submitted text (whitespace-trimmed).")
    created_at: str = Field(description="Submission time, ISO 8601 UTC.")
    started_at: str | None = Field(description="Time the analysis started; `null` while queued.")
    finished_at: str | None = Field(description="Time the analysis ended; `null` until then.")
    queue_position: int | None = Field(
        description="1-based position in the queue while `queued`, otherwise `null`."
    )
    results: list[ClaimResult] | None = Field(
        description=(
            "One entry per check-worthy claim when `completed` (empty list when "
            "the text has no check-worthy claims), otherwise `null`."
        )
    )
    error: str | None = Field(
        description="Short error message when `failed`, otherwise `null`."
    )


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]] = Field(
        description=(
            "Human-readable error message. For a malformed request body it is "
            "a list of validation errors instead."
        ),
        examples=["Text is too long: the limit is 20000 characters."],
    )


class HealthResponse(BaseModel):
    pipeline: Literal["initializing", "ready"]
    last_error: str | None
    searxng: bool
    ollama: bool
