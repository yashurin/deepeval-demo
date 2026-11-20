"""Gradio page: one prompt, one local answer, then DeepEval scores.

    docker compose up --build deepeval-ui

On the host, with Ollama already running:

    python -m app.ui

Open http://localhost:7860. Gradio also binds 0.0.0.0 inside Docker so
the published port works. Browsers cannot open http://0.0.0.0. Each
metric is another local generation, so a click can take several
minutes on CPU. One click runs at a time.
"""

from __future__ import annotations

import os
import re
import threading

# Set before Gradio is imported so the page does not send usage stats.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from app.config import Settings, load_settings, prepare_process, resolved_judge

prepare_process()

import gradio as gr
from deepeval.test_case import LLMTestCase

from app.evaluate import build_metrics, normalize_metric_names
from app.judge import LocalOllamaJudge
from app.ollama_client import OllamaError, ensure_model, generate
from app.support_bot import answer_from_context

# Same shape as the plain questions in the batch runner.
_PLAIN_SYSTEM = (
    "Answer the question in one short sentence. "
    "Do not explain your reasoning."
)

_METRIC_CHOICES = [
    "relevancy",
    "faithfulness",
    "correctness",
    "helpfulness",
    "contextual_relevancy",
]
_CONTEXT_METRICS = {"faithfulness", "contextual_relevancy"}

_lock = threading.Lock()
_judge: LocalOllamaJudge | None = None


def parse_context(text: str) -> list[str]:
    """Split a context box on blank lines. One block is one excerpt."""
    return [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]


def _prepare(settings: Settings) -> LocalOllamaJudge:
    global _judge
    if _judge is None:
        ensure_model(settings)
        _judge = LocalOllamaJudge(settings)
    return _judge


def _format_metric(metric) -> str:
    name = getattr(metric, "name", None) or metric.__class__.__name__
    error = getattr(metric, "error", None)
    if error:
        return f"### {name}\n- **Result:** error\n- **Detail:** {error}"
    score = getattr(metric, "score", None)
    score_text = "n/a" if score is None else f"{float(score):.2f}"
    threshold = getattr(metric, "threshold", None)
    threshold_text = "" if threshold is None else f" (threshold {float(threshold):.2f})"
    success = getattr(metric, "success", None)
    if success is None and hasattr(metric, "is_successful"):
        try:
            success = bool(metric.is_successful())
        except Exception:
            success = False
    status = "pass" if success is True else "fail"
    reason = getattr(metric, "reason", None) or ""
    lines = [
        f"### {name}",
        f"- **Result:** {status}",
        f"- **Score:** {score_text}{threshold_text}",
    ]
    if reason:
        lines.append(f"- **Reason:** {reason}")
    return "\n".join(lines)


def evaluate_prompt(
    prompt: str,
    context_text: str,
    expected: str,
    metric_names: list[str] | None,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Generate one answer and score it. Returns (answer, markdown report)."""
    settings = settings or load_settings()
    question = (prompt or "").strip()
    if not question:
        return "", "Enter a prompt."

    selected = normalize_metric_names(",".join(metric_names or []))
    if not selected:
        return "", "Pick at least one metric."

    chunks = parse_context(context_text)
    expected_output = (expected or "").strip()
    notes: list[str] = []
    runnable: list[str] = []
    for name in selected:
        if name in _CONTEXT_METRICS and not chunks:
            notes.append(
                f"`{name}` needs retrieval context, so it was skipped."
            )
            continue
        if name == "correctness" and not expected_output:
            notes.append(
                "`correctness` needs an expected answer, so it was skipped."
            )
            continue
        runnable.append(name)

    if not _lock.acquire(blocking=False):
        return "", "Another evaluation is still running. Wait for it to finish."
    try:
        try:
            judge = _prepare(settings)
            if chunks:
                answer = answer_from_context(settings, question, chunks)
            else:
                answer = generate(settings, system=_PLAIN_SYSTEM, user=question)
        except (OllamaError, OSError) as exc:
            return "", f"Generation failed: {exc}"

        sections = [
            f"Model `{settings.model}`. Judge `{resolved_judge(settings)}`.",
        ]
        if chunks:
            sections.append(
                "The model saw these excerpts:\n\n"
                + "\n".join(f"- {chunk}" for chunk in chunks)
            )
        else:
            sections.append("No retrieval context. This was a plain answer.")
        if notes:
            sections.append("\n".join(notes))
        if not runnable:
            return answer, "\n\n".join(sections)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_output or None,
            retrieval_context=chunks or None,
            name="ui",
        )
        try:
            metrics = build_metrics(
                settings, judge, ",".join(runnable), verbose=False
            )
        except SystemExit as exc:
            return answer, f"Could not build metrics: {exc}"

        reports = []
        for metric in metrics:
            try:
                metric.measure(test_case)
            except Exception as exc:
                name = getattr(metric, "name", None) or metric.__class__.__name__
                reports.append(
                    f"### {name}\n- **Result:** error\n- **Detail:** {exc}"
                )
                continue
            reports.append(_format_metric(metric))
        sections.extend(reports)
        return answer, "\n\n".join(sections)
    finally:
        _lock.release()


def build_demo(settings: Settings | None = None) -> gr.Blocks:
    settings = settings or load_settings()
    intro = (
        f"Local DeepEval. The bot and the judge are `{settings.model}` "
        f"at `{settings.ollama_base_url}`.\n\n"
        "Leave context empty for a plain answer. Paste one policy excerpt "
        "per paragraph to score faithfulness and contextual relevancy. "
        "Correctness also needs an expected answer.\n\n"
        "On a CPU, each selected metric can take several minutes."
    )

    with gr.Blocks(title="Local DeepEval") as demo:
        gr.Markdown(intro)
        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="What is your return policy if the shoes don't fit?",
        )
        context = gr.Textbox(
            label="Retrieval context (optional)",
            lines=5,
            placeholder=(
                "Customers may return unworn items within 30 days of "
                "delivery for a full refund. Return shipping is free."
            ),
        )
        expected = gr.Textbox(
            label="Expected answer (optional, used by correctness)",
            lines=2,
            placeholder=(
                "Unworn items can be returned within 30 days of delivery "
                "for a full refund, and return shipping is free."
            ),
        )
        metrics = gr.CheckboxGroup(
            choices=_METRIC_CHOICES,
            value=["relevancy", "helpfulness"],
            label="Metrics",
        )
        button = gr.Button("Generate and evaluate", variant="primary")
        answer = gr.Textbox(label="Model answer", lines=4)
        report = gr.Markdown(label="Scores")

        def on_submit(prompt, context, expected, metrics):
            return evaluate_prompt(prompt, context, expected, metrics, settings)

        button.click(
            on_submit,
            inputs=[prompt, context, expected, metrics],
            outputs=[answer, report],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    # Inside the container Gradio listens on GRADIO_SERVER_PORT and on
    # every interface. UI_PORT is the port published on the host. Those
    # match unless you remap the port in Compose.
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    public_port = os.environ.get("UI_PORT", str(port))
    demo = build_demo()
    print(
        f"\nOpen http://localhost:{public_port}\n"
        "Browsers cannot open http://0.0.0.0. Use localhost and the "
        "published port.\n",
        flush=True,
    )
    # queue is on so the button shows progress. share stays off: this
    # demo does not open a public Gradio tunnel. quiet hides Gradio's
    # own line, which prints the 0.0.0.0 bind address.
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
