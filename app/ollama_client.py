"""Talk to the local Ollama server over HTTP.

This module is the model under test. DeepEval's OllamaModel (see judge.py)
is a separate client that scores whatever this module returns. Both use
the model name from Settings, so the app and the judge stay in lockstep.
"""

from __future__ import annotations

import json
import re
import time

import requests

from app.config import Settings

# qwen3-family templates sometimes still wrap a draft in these tags.
# The judge should see the answer, not the draft.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """The local Ollama server rejected a call or never became ready."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def is_retryable(exc: BaseException) -> bool:
    """True for a dropped connection, a 5xx, or a flaky local completion.

    A 4xx is the request, not a blip, so it is not retried. The think-flag
    fallback in ``generate`` handles the one 400 we know how to repair.
    """
    if isinstance(exc, OllamaError):
        return exc.retryable
    status = _status_code(exc)
    if status is not None:
        return status >= 500
    if isinstance(exc, ValueError):
        # Empty or invalid JSON. A small local model often gets the
        # shape right on the next try.
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "connect" in name:
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "server disconnected",
        )
    )


def retry_call(retries: int, label: str, fn):
    """Call ``fn`` once, then up to ``retries`` more times on a transient error."""
    attempts = max(0, retries) + 1
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = min(8, 2 ** (attempt - 1))
            detail = str(exc).strip().splitlines()[0][:300]
            print(
                f"{label} failed ({detail}). "
                f"Retrying in {delay}s ({attempt}/{retries}).",
                flush=True,
            )
            time.sleep(delay)
    assert last is not None
    raise last


def strip_thinking(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text or "")
    return cleaned.strip()


def wait_until_ready(settings: Settings, attempts: int = 60, pause: float = 2.0) -> None:
    """Block until Ollama answers /api/tags.

    Compose already waits for the healthcheck. This second check covers
    `python -m app.evaluate` on the host, where nothing else waited.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    last_error = "no response"
    for _ in range(attempts):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(pause)
    raise OllamaError(
        f"Ollama at {settings.ollama_base_url} did not become ready ({last_error})."
    )


def _model_is_present(settings: Settings) -> bool:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    names = {item.get("name") for item in response.json().get("models", [])}
    # `ollama list` shows either "qwen3.5:2b" or "qwen3.5:2b" with an
    # implicit ":latest". Accept both forms.
    wanted = settings.model
    if ":" not in wanted:
        wanted = f"{wanted}:latest"
    return wanted in names or settings.model in names


def unload_model(settings: Settings) -> None:
    """Ask Ollama to drop this model's weights.

    Comparison mode does this so two small models are not resident at
    the same time. A failure here is not fatal: the next call still runs.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    try:
        response = requests.post(
            url,
            json={"model": settings.model, "keep_alive": 0},
            timeout=30,
        )
    except requests.RequestException as exc:
        print(f"Could not unload {settings.model} ({exc}). Continuing.", flush=True)
        return
    if response.status_code >= 400:
        print(
            f"Could not unload {settings.model} "
            f"({response.status_code}). Continuing.",
            flush=True,
        )


def ensure_model(settings: Settings) -> None:
    """Pull the configured model if the volume does not have it yet."""
    wait_until_ready(settings)
    if _model_is_present(settings):
        print(f"Model {settings.model} is already installed.", flush=True)
        return

    print(
        f"Pulling {settings.model}. The first download is a few GB and "
        "is kept in the ollama_models volume.",
        flush=True,
    )
    url = f"{settings.ollama_base_url.rstrip('/')}/api/pull"
    last_percent = -10
    with requests.post(
        url,
        json={"name": settings.model, "stream": True},
        stream=True,
        timeout=(10, settings.request_timeout),
    ) as response:
        if response.status_code >= 400:
            raise OllamaError(
                f"Pull failed ({response.status_code}): {response.text.strip()}"
            )
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event.get("error"):
                raise OllamaError(f"Pull failed: {event['error']}")
            total = event.get("total") or 0
            completed = event.get("completed") or 0
            if total and completed:
                percent = int(completed * 100 / total)
                if percent >= last_percent + 10:
                    print(f"  downloading {settings.model}: {percent}%", flush=True)
                    last_percent = percent
            elif event.get("status") and event["status"] != "success":
                # Manifest and unpack lines. Skip the noisy byte updates.
                status = event["status"]
                if "pulling" in status or status.startswith("verifying"):
                    print(f"  {status}", flush=True)
    print(f"Model {settings.model} is ready.", flush=True)


def generate(settings: Settings, *, system: str, user: str) -> str:
    """Ask the local model for one completion and return the answer text.

    A connection error, HTTP 5xx, or empty completion is tried again.
    ``OLLAMA_RETRIES`` is the number of extra attempts (default 2).
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload: dict = {
        "model": settings.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": settings.app_temperature,
            "num_ctx": settings.num_ctx,
        },
    }
    if settings.think is not None:
        payload["think"] = settings.think
    # Flipped off for the rest of this call when the server rejects `think`.
    send_think = settings.think is not None

    def once() -> str:
        nonlocal send_think
        body = dict(payload)
        if not send_think:
            body.pop("think", None)
        response = _post_chat(url, body, settings)
        if response.status_code == 400 and send_think and _rejected_think(response):
            # A non-thinking model returns 400 when `think` is present.
            print(
                "This model rejected the think flag. Retrying without it. "
                "Set OLLAMA_THINK=omit to skip the flag next time.",
                flush=True,
            )
            send_think = False
            body.pop("think", None)
            response = _post_chat(url, body, settings)

        if response.status_code >= 500:
            raise OllamaError(
                f"Chat failed ({response.status_code}): {response.text.strip()}",
                retryable=True,
            )
        if response.status_code >= 400:
            raise OllamaError(
                f"Chat failed ({response.status_code}): {response.text.strip()}"
            )
        try:
            content = response.json().get("message", {}).get("content") or ""
        except ValueError as exc:
            raise OllamaError(
                f"Ollama returned a body that is not JSON: {exc}",
                retryable=True,
            ) from exc
        answer = strip_thinking(content)
        if not answer:
            raise OllamaError("The model returned an empty answer.", retryable=True)
        return answer

    return retry_call(settings.retries, "Bot call", once)


def _post_chat(url: str, payload: dict, settings: Settings) -> requests.Response:
    try:
        return requests.post(url, json=payload, timeout=settings.request_timeout)
    except requests.RequestException as exc:
        raise OllamaError(
            f"Could not reach Ollama at {url}: {exc}",
            retryable=True,
        ) from exc


def _rejected_think(response: requests.Response) -> bool:
    return "think" in response.text.lower()
