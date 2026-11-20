"""Ask the local Ollama model for an answer to each sample question.

Plain questions go out as-is. RAG questions include the policy excerpts
stored on the golden. Those same excerpts become ``retrieval_context``
on the ``LLMTestCase``, which is what faithfulness and contextual
relevancy read.

    python -m app.generate
"""

from __future__ import annotations

from app.config import Settings, prepare_process

prepare_process()

from deepeval.test_case import LLMTestCase

from app.ollama_client import generate
from app.sample_data import GOLDENS, Golden, case_input
from app.support_bot import answer_from_context

# Plain questions have no policy to lean on. One sentence keeps the
# later correctness comparison easy to read.
_QA_SYSTEM = (
    "Answer the question in one short sentence. "
    "Do not explain your reasoning."
)


def build_test_cases(
    settings: Settings, goldens: list[Golden] | None = None
) -> list[LLMTestCase]:
    """Call the model once per golden and wrap each turn in an LLMTestCase."""
    cases: list[LLMTestCase] = []
    for golden in GOLDENS if goldens is None else goldens:
        print(f"\n[{golden.name}]", flush=True)
        print(f"Question: {golden.question}", flush=True)
        if golden.history:
            print(f"Prior turns: {len(golden.history)}", flush=True)
        if golden.retrieval_context:
            context = list(golden.retrieval_context)
            print("Retrieved policy:", flush=True)
            for chunk in context:
                print(f"  - {chunk}", flush=True)
            actual_output = answer_from_context(
                settings,
                golden.question,
                context,
                history=golden.history,
            )
        else:
            context = None
            print("Retrieved policy: (none, plain question)", flush=True)
            user = case_input(golden)
            if golden.history:
                user = f"{user}\n\nAnswer the last customer message."
            actual_output = generate(settings, system=_QA_SYSTEM, user=user)
        print(f"Answer:   {actual_output}", flush=True)
        cases.append(
            LLMTestCase(
                input=case_input(golden),
                actual_output=actual_output,
                expected_output=golden.expected_output,
                retrieval_context=context,
                name=golden.name,
            )
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    """Print answers for the sample set. Does not score them."""
    import argparse
    import sys

    from app.config import load_settings
    from app.ollama_client import OllamaError, ensure_model

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    settings = load_settings()
    print(f"Ollama: {settings.ollama_base_url}", flush=True)
    print(f"Model:  {settings.model}", flush=True)
    try:
        ensure_model(settings)
        build_test_cases(settings)
    except (OllamaError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
