"""Download Ollama models into the local volume and exit.

    python -m app.pull
    python -m app.pull phi4-mini

With no arguments this pulls ``OLLAMA_MODEL``. ``docker compose up`` does
that pull itself, before the quick evaluation, the first time the volume
is empty.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from app.config import load_settings
from app.ollama_client import OllamaError, ensure_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        help="Tags to pull. The default is OLLAMA_MODEL.",
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    names = args.models or [settings.model]
    print(f"Ollama: {settings.ollama_base_url}", flush=True)
    try:
        for name in names:
            ensure_model(replace(settings, model=name))
    except (OllamaError, OSError) as exc:
        print(f"\nPull failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
