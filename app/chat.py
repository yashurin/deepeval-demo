"""Ask the support bot one question, without running metrics.

    python -m app.chat "What is your return policy?"
"""

from __future__ import annotations

import argparse
import sys

from app.config import load_settings
from app.ollama_client import OllamaError, ensure_model
from app.support_bot import answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Question for the support bot.")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question given.", file=sys.stderr)
        return 1

    settings = load_settings()
    try:
        ensure_model(settings)
        text, context = answer(settings, question)
    except (OllamaError, OSError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    print(text)
    print("\nRetrieved policy:")
    for chunk in context:
        print(f"- {chunk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
