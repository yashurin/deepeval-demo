"""Score model answers with DeepEval.

The judge is ``deepeval.models.OllamaModel`` (see ``app.judge``), pointed
at the same local model that wrote the answers. Nothing here calls a
hosted API.

The command line lives in ``app.run_evals``. ``python -m app.evaluate``
still works and calls that module.
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime, timezone

from app.config import Settings, prepare_process

prepare_process()

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.judge import LocalOllamaJudge

# These metrics read retrieval_context. Plain questions omit that field,
# so run_evals scores them with the other metrics only.
CONTEXT_METRICS = frozenset({"faithfulness", "contextual_relevancy"})

_METRIC_HELP = (
    "relevancy, faithfulness, correctness, helpfulness, contextual_relevancy"
)


def normalize_metric_names(names: str) -> list[str]:
    """Split a comma-separated metric list. Hyphens and underscores match."""
    return [
        part.strip().lower().replace("-", "_")
        for part in names.split(",")
        if part.strip()
    ]


def build_metrics(
    settings: Settings,
    judge: LocalOllamaJudge,
    names: str,
    *,
    verbose: bool = True,
):
    """Build the metrics named in a comma-separated list."""
    selected = normalize_metric_names(names)
    if not selected:
        raise SystemExit(f"METRICS is empty. Use {_METRIC_HELP}.")

    known = {
        "relevancy": lambda: AnswerRelevancyMetric(
            threshold=settings.answer_relevancy_threshold,
            model=judge,
            include_reason=True,
            async_mode=settings.run_async,
            verbose_mode=verbose,
        ),
        "faithfulness": lambda: FaithfulnessMetric(
            threshold=settings.faithfulness_threshold,
            model=judge,
            include_reason=True,
            async_mode=settings.run_async,
            verbose_mode=verbose,
        ),
        "correctness": lambda: GEval(
            name="Correctness",
            criteria=(
                "Decide whether the actual output has the same meaning as "
                "the expected output."
            ),
            # Fixed steps skip an extra judge call that would otherwise
            # invent the steps from the criteria on every run.
            evaluation_steps=[
                "List the facts stated in the expected output.",
                "Check that the actual output states those facts.",
                "Treat a contradiction of the expected output as a failure.",
                "Allow different wording when the meaning matches.",
            ],
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=settings.correctness_threshold,
            model=judge,
            async_mode=settings.run_async,
            verbose_mode=verbose,
        ),
        "helpfulness": lambda: GEval(
            name="Helpfulness",
            criteria=(
                "Decide whether the answer is helpful to the person who "
                "asked. A helpful answer is specific, on topic, and short "
                "enough to use. Saying you do not know is helpful when "
                "that is the honest reply. A vague, padded, or off-topic "
                "answer is not helpful."
            ),
            evaluation_steps=[
                "Read the question and name what the person needs.",
                "Check that the answer speaks to that need.",
                "Treat a direct, specific answer as more helpful than a vague one.",
                "Treat an honest admission of not knowing as helpful.",
                "Treat filler, a different topic, or a buried answer as unhelpful.",
            ],
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
            ],
            threshold=settings.helpfulness_threshold,
            model=judge,
            async_mode=settings.run_async,
            verbose_mode=verbose,
        ),
        "contextual_relevancy": lambda: ContextualRelevancyMetric(
            threshold=settings.contextual_relevancy_threshold,
            model=judge,
            include_reason=True,
            async_mode=settings.run_async,
            verbose_mode=verbose,
        ),
    }
    # Friendly aliases for the names above.
    known["answer_relevancy"] = known["relevancy"]
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise SystemExit(
            f"Unknown metric(s): {', '.join(unknown)}. Choose from {_METRIC_HELP}."
        )
    return [known[name]() for name in selected]


def score(
    settings: Settings,
    judge: LocalOllamaJudge,
    test_cases: list[LLMTestCase],
    metric_names: str,
    judge_model: str | None = None,
):
    """Run deepeval.evaluate. One local model, so concurrency stays at 1."""
    metrics = build_metrics(settings, judge, metric_names)
    os.makedirs(settings.results_dir, exist_ok=True)
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "model": settings.model,
            "judge_model": judge_model or settings.model,
            "ollama_base_url": settings.ollama_base_url,
        },
        identifier="local-ollama-demo",
        async_config=AsyncConfig(
            run_async=settings.run_async,
            max_concurrent=1,
        ),
        display_config=DisplayConfig(
            show_indicator=True,
            print_results=True,
            verbose_mode=True,
            inspect_after_run=False,
            results_folder=settings.results_dir,
        ),
        error_config=ErrorConfig(ignore_errors=settings.ignore_errors),
    )


def _field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _print_summary(result, title: str = "Summary") -> bool:
    """Print score, reason, and pass/fail for each metric.

    Return True when every metric passed.
    """
    all_passed = True
    print(f"\n{title}", flush=True)
    for test_result in _field(result, "test_results") or []:
        label = _field(test_result, "name") or _field(test_result, "input") or "case"
        for metric in _field(test_result, "metrics_data") or []:
            name = _field(metric, "name", "metric")
            error = _field(metric, "error")
            if error:
                all_passed = False
                print(f"  {label} / {name}: ERROR {error}", flush=True)
                continue
            score_value = _field(metric, "score")
            score_text = "n/a" if score_value is None else f"{float(score_value):.2f}"
            success = _field(metric, "success") is True
            if not success:
                all_passed = False
            status = "pass" if success else "fail"
            threshold = _field(metric, "threshold")
            print(
                f"  {label} / {name}: {status} "
                f"(score {score_text}, threshold {threshold})",
                flush=True,
            )
            reason = _field(metric, "reason")
            if reason:
                print(
                    textwrap.fill(
                        f"reason: {reason}",
                        width=88,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    ),
                    flush=True,
                )
    return all_passed


def collect_rows(result, model: str) -> list[dict]:
    """Flatten one DeepEval result into rows for the table and JSON file."""
    rows = []
    for test_result in _field(result, "test_results") or []:
        label = _field(test_result, "name") or _field(test_result, "input") or "case"
        for metric in _field(test_result, "metrics_data") or []:
            error = _field(metric, "error") or None
            score_value = _field(metric, "score")
            threshold = _field(metric, "threshold")
            success = _field(metric, "success") is True and not error
            rows.append(
                {
                    "case": label,
                    "metric": _field(metric, "name", "metric"),
                    "model": model,
                    "score": None if score_value is None else float(score_value),
                    "threshold": None if threshold is None else float(threshold),
                    "passed": success,
                    "reason": _field(metric, "reason") or None,
                    "error": error,
                }
            )
    return rows


def _score_cell(row: dict | None) -> str:
    if row is None:
        return "-"
    if row["error"]:
        return "error"
    if row["score"] is None:
        return "n/a"
    status = "pass" if row["passed"] else "fail"
    return f"{row['score']:.2f} {status}"


def format_score_table(rows: list[dict], models: list[str]) -> str:
    """One row per case and metric, one column per model under test."""
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["case"], row["metric"])
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        grouped[key][row["model"]] = row

    case_width = max([len("case"), *(len(case) for case, _ in order)] or [4])
    metric_width = max(
        [len("metric"), *(len(metric) for _, metric in order)] or [6]
    )
    model_widths = []
    for model in models:
        cells = [_score_cell(grouped[key].get(model)) for key in order]
        model_widths.append(
            max([len(model), *(len(cell) for cell in cells)] or [len(model)])
        )

    def line(case: str, metric: str, cells: list[str]) -> str:
        parts = [case.ljust(case_width), metric.ljust(metric_width)]
        parts.extend(cell.ljust(width) for cell, width in zip(cells, model_widths))
        return "  ".join(parts)

    header = line("case", "metric", models)
    rule = line(
        "-" * case_width,
        "-" * metric_width,
        ["-" * width for width in model_widths],
    )
    body = [
        line(
            case,
            metric,
            [_score_cell(grouped[(case, metric)].get(model)) for model in models],
        )
        for case, metric in order
    ]
    if not body:
        return "Score table\n  (no scores)"
    return "\n".join(["Score table", header, rule, *body])


def print_score_table(rows: list[dict], models: list[str]) -> None:
    print(f"\n{format_score_table(rows, models)}", flush=True)


def summary_payload(
    *,
    ollama_base_url: str,
    judge_model: str,
    models: list[str],
    metrics: list[str],
    rows: list[dict],
    passed: bool,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ollama_base_url": ollama_base_url,
        "judge_model": judge_model,
        "models": models,
        "metrics": metrics,
        "passed": passed,
        "rows": rows,
    }


def write_summary(path: str, payload: dict) -> None:
    """Write the score table's data as JSON. Parent folders are created."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Same entry as ``python -m app.run_evals``."""
    from app.run_evals import main as run_main

    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
