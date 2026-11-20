"""Generate answers with the local model, then score them with DeepEval.

Examples, from the app container:

    python -m app.run_evals --quick
    python -m app.run_evals
    python -m app.run_evals --compare phi4-mini --quick
    python -m app.evaluate --fail-on-threshold

``app.evaluate`` calls this module, so either command works.
The judge is a local Ollama model. No hosted API is used.

``--compare`` runs the same questions through a second model. The judge
stays ``OLLAMA_MODEL`` (or ``JUDGE_MODEL`` when that is set) so the two
score columns share one scorer.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

from app.config import Settings, load_settings, prepare_process, resolved_judge

prepare_process()

from app.evaluate import (
    CONTEXT_METRICS,
    _print_summary,
    collect_rows,
    normalize_metric_names,
    print_score_table,
    score,
    summary_payload,
    write_summary,
)
from app.generate import build_test_cases
from app.judge import LocalOllamaJudge
from app.ollama_client import OllamaError, ensure_model, unload_model
from app.sample_data import GOLDENS, Golden


def _quick_goldens() -> list[Golden]:
    for golden in GOLDENS:
        if golden.name == "returns":
            return [golden]
    return list(GOLDENS[:1])


def models_under_test(primary: str, compare: str | None) -> list[str]:
    """Return the model tags that will answer the questions."""
    if not compare:
        return [primary]
    if compare == primary:
        raise SystemExit(
            f"{compare} is already OLLAMA_MODEL. "
            "Pass a different tag to --compare."
        )
    return [primary, compare]


def _evaluate_cases(
    settings: Settings,
    judge: LocalOllamaJudge,
    test_cases,
    selected: list[str],
    model_name: str,
    judge_model: str,
) -> tuple[bool, list[dict]]:
    context_names = [name for name in selected if name in CONTEXT_METRICS]
    other_names = [name for name in selected if name not in CONTEXT_METRICS]
    rag_cases = [case for case in test_cases if case.retrieval_context]
    plain_cases = [case for case in test_cases if not case.retrieval_context]

    passed = True
    rows: list[dict] = []
    if other_names:
        result = score(
            settings,
            judge,
            test_cases,
            ",".join(other_names),
            judge_model=judge_model,
        )
        rows.extend(collect_rows(result, model_name))
        passed = _print_summary(
            result,
            title=f"Summary ({model_name}: {', '.join(other_names)})",
        ) and passed
    elif plain_cases and context_names:
        print(
            f"\nSkipping {len(plain_cases)} plain question(s). "
            f"{', '.join(context_names)} need retrieval context.",
            flush=True,
        )

    if context_names:
        if not rag_cases:
            verb = "were" if len(context_names) > 1 else "was"
            print(
                "\nNo RAG cases in this run, so "
                f"{', '.join(context_names)} {verb} skipped.",
                flush=True,
            )
        else:
            if plain_cases:
                print(
                    f"\nScoring {', '.join(context_names)} on "
                    f"{len(rag_cases)} case(s) that include retrieval context.",
                    flush=True,
                )
            result = score(
                settings,
                judge,
                rag_cases,
                ",".join(context_names),
                judge_model=judge_model,
            )
            rows.extend(collect_rows(result, model_name))
            passed = _print_summary(
                result,
                title=f"Summary ({model_name}: {', '.join(context_names)})",
            ) and passed
    return passed, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Score only the returns question, and only with correctness.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit 1 when a metric misses its threshold or errors.",
    )
    parser.add_argument(
        "--compare",
        metavar="MODEL",
        default=None,
        help=(
            "Also answer every question with this second Ollama model "
            "and print a side-by-side score table. "
            "Defaults to COMPARE_MODEL when set."
        ),
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="JSON summary path. Default: <RESULTS_DIR>/summary.json",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    goldens = _quick_goldens() if args.quick else list(GOLDENS)
    metric_names = "correctness" if args.quick else settings.metrics
    selected = normalize_metric_names(metric_names)
    compare = args.compare if args.compare is not None else settings.compare_model
    under_test = models_under_test(settings.model, compare)
    judge_model = resolved_judge(settings)
    json_path = args.json or os.path.join(settings.results_dir, "summary.json")

    print("Local DeepEval demo", flush=True)
    print(f"  Ollama:           {settings.ollama_base_url}", flush=True)
    print(f"  Model under test: {', '.join(under_test)}", flush=True)
    print(f"  Judge:            {judge_model} (deepeval.models.OllamaModel)", flush=True)
    print(f"  Metrics:          {', '.join(selected)}", flush=True)
    print(f"  Retries:          {settings.retries} extra attempt(s)", flush=True)
    print(f"  JSON summary:     {json_path}", flush=True)
    if len(under_test) > 1:
        print(
            "Comparison runs the questions twice and pulls any missing model. "
            "The judge stays "
            f"{judge_model}.",
            flush=True,
        )
    print(
        "A small model on CPU can take several minutes per metric.",
        flush=True,
    )

    try:
        to_pull = []
        for name in [judge_model, *under_test]:
            if name not in to_pull:
                to_pull.append(name)
        for name in to_pull:
            ensure_model(replace(settings, model=name))
        judge = LocalOllamaJudge(replace(settings, model=judge_model))
    except (OllamaError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    passed = True
    rows: list[dict] = []
    try:
        for index, model_name in enumerate(under_test):
            # Drop whichever weights are not needed for the next step so
            # two small models are not resident together.
            if model_name != judge_model:
                unload_model(replace(settings, model=judge_model))
            print(f"\n=== Answers from {model_name} ===", flush=True)
            test_cases = build_test_cases(
                replace(settings, model=model_name), goldens
            )
            if model_name != judge_model:
                unload_model(replace(settings, model=model_name))
            model_passed, model_rows = _evaluate_cases(
                replace(settings, model=model_name),
                judge,
                test_cases,
                selected,
                model_name,
                judge_model,
            )
            passed = model_passed and passed
            rows.extend(model_rows)
            if index + 1 < len(under_test) and model_name == judge_model:
                unload_model(replace(settings, model=model_name))
    except (OllamaError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    print_score_table(rows, under_test)
    payload = summary_payload(
        ollama_base_url=settings.ollama_base_url,
        judge_model=judge_model,
        models=under_test,
        metrics=selected,
        rows=rows,
        passed=passed,
    )
    write_summary(json_path, payload)
    print(f"\nJSON summary: {json_path}", flush=True)
    print(f"DeepEval report folder: {settings.results_dir}", flush=True)
    print(f"Overall: {'pass' if passed else 'fail'}", flush=True)
    if args.fail_on_threshold and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
