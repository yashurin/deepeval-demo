"""Read the demo settings from the environment.

Docker Compose injects these. On the host, values come from a .env file
(see .env.example) or from the shell. Existing environment variables win
over the file, which is python-dotenv's default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def prepare_process() -> None:
    """Load .env, then set the switches a local DeepEval run needs.

    Call this before importing deepeval. ``setdefault`` keeps a value that
    Docker or the shell already provided.
    """
    load_dotenv()
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    os.environ.setdefault("DEEPEVAL_NO_INSPECT_PROMPT", "1")
    os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _float(name: str, default: float) -> float:
    value = _raw(name)
    return default if value is None else float(value)


def _int(name: str, default: int) -> int:
    value = _raw(name)
    return default if value is None else int(value)


def _bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true or false (got {value!r}).")


def _think(name: str, default: bool | None) -> bool | None:
    """true/false, or 'omit' to leave the flag off the Ollama request."""
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"omit", "auto", "default"}:
        return None
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true, false, or omit (got {value!r}).")


@dataclass(frozen=True)
class Settings:
    # One model name for both roles. The bot calls Ollama over HTTP.
    # The judge is deepeval.models.OllamaModel pointed at the same tag.
    ollama_base_url: str
    model: str
    think: bool | None
    app_temperature: float
    judge_temperature: float
    num_ctx: int
    request_timeout: float
    answer_relevancy_threshold: float
    faithfulness_threshold: float
    correctness_threshold: float
    helpfulness_threshold: float
    contextual_relevancy_threshold: float
    # Extra attempts after the first try, for a dropped connection,
    # a 5xx, or an empty completion. 0 means try once.
    retries: int
    metrics: str
    run_async: bool
    ignore_errors: bool
    results_dir: str
    # Optional second model for comparison mode. The judge stays `model`
    # unless judge_model is set, so the two columns share one scorer.
    compare_model: str | None
    judge_model: str | None


def load_settings() -> Settings:
    return Settings(
        ollama_base_url=_raw("OLLAMA_BASE_URL") or "http://localhost:11434",
        model=_raw("OLLAMA_MODEL") or "qwen3.5:2b",
        think=_think("OLLAMA_THINK", False),
        app_temperature=_float("APP_TEMPERATURE", 0.2),
        judge_temperature=_float("JUDGE_TEMPERATURE", 0.0),
        num_ctx=_int("NUM_CTX", 8192),
        request_timeout=_float("OLLAMA_TIMEOUT", 600),
        answer_relevancy_threshold=_float("ANSWER_RELEVANCY_THRESHOLD", 0.5),
        faithfulness_threshold=_float("FAITHFULNESS_THRESHOLD", 0.5),
        correctness_threshold=_float("CORRECTNESS_THRESHOLD", 0.5),
        helpfulness_threshold=_float("HELPFULNESS_THRESHOLD", 0.5),
        contextual_relevancy_threshold=_float("CONTEXTUAL_RELEVANCY_THRESHOLD", 0.5),
        retries=max(0, _int("OLLAMA_RETRIES", 2)),
        metrics=(
            _raw("METRICS")
            or "relevancy,faithfulness,correctness,helpfulness,contextual_relevancy"
        ),
        run_async=_bool("RUN_ASYNC", False),
        ignore_errors=_bool("IGNORE_EVAL_ERRORS", True),
        results_dir=_raw("RESULTS_DIR") or "results",
        compare_model=_raw("COMPARE_MODEL"),
        judge_model=_raw("JUDGE_MODEL"),
    )


def resolved_judge(settings: Settings) -> str:
    """Model tag DeepEval uses as the judge."""
    return settings.judge_model or settings.model
