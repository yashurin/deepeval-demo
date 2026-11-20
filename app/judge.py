"""The LLM-as-a-judge: DeepEval's OllamaModel, pointed at the local model.

Metrics call ``generate(prompt, schema=...)``. OllamaModel puts
``generation_kwargs`` in Ollama's ``options`` dict (temperature, num_ctx).
``think`` is not an option. It is its own chat argument, and qwen3.5 turns
thinking on unless the request says ``think=false``. A thinking trace in
the middle of the JSON schema makes the metric fail, so this subclass
sends the flag itself.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Tuple, Union

from pydantic import BaseModel

from deepeval.models import OllamaModel

from app.config import Settings
from app.ollama_client import is_retryable, retry_call, strip_thinking

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LocalOllamaJudge(OllamaModel):
    """OllamaModel that can disable thinking on every judge call."""

    def __init__(self, settings: Settings):
        super().__init__(
            model=settings.model,
            base_url=settings.ollama_base_url,
            temperature=settings.judge_temperature,
            # Forwarded to Ollama as options. Caps RAM used by the context.
            generation_kwargs={"num_ctx": settings.num_ctx},
            # httpx timeout on the client DeepEval builds.
            timeout=settings.request_timeout,
        )
        self.think = settings.think
        self.retries = settings.retries

    def generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio=None,
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=False)

        def once():
            messages = self._prompt_messages(prompt, audio)
            request = self._request(messages, schema)
            response = self._chat(client.chat, request)
            return self._parse(response, schema)

        return retry_call(self.retries, "Judge call", once), 0

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio=None,
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        attempts = self.retries + 1
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                messages = self._prompt_messages(prompt, audio)
                request = self._request(messages, schema)
                response = await self._achat(client.chat, request)
                return self._parse(response, schema), 0
            except Exception as exc:
                last = exc
                if attempt >= attempts or not is_retryable(exc):
                    raise
                delay = min(8, 2 ** (attempt - 1))
                detail = str(exc).strip().splitlines()[0][:300]
                print(
                    f"Judge call failed ({detail}). "
                    f"Retrying in {delay}s ({attempt}/{self.retries}).",
                    flush=True,
                )
                await asyncio.sleep(delay)
        assert last is not None
        raise last

    def _request(self, messages, schema) -> dict:
        request = {
            "model": self.name,
            "messages": messages,
            # A Pydantic schema becomes Ollama's structured-output grammar,
            # so the judge has to answer with that JSON shape.
            "format": schema.model_json_schema() if schema else None,
            "options": {
                "temperature": self.temperature,
                **self.generation_kwargs,
            },
        }
        if self.think is not None:
            request["think"] = self.think
        return request

    def _chat(self, chat, request):
        try:
            return chat(**request)
        except Exception as exc:
            if not self._think_was_rejected(exc):
                raise
            request.pop("think", None)
            return chat(**request)

    async def _achat(self, chat, request):
        try:
            return await chat(**request)
        except Exception as exc:
            if not self._think_was_rejected(exc):
                raise
            request.pop("think", None)
            return await chat(**request)

    def _think_was_rejected(self, exc: Exception) -> bool:
        if self.think is None or "think" not in str(exc).lower():
            return False
        print(
            "Judge call rejected the think flag. Retrying without it. "
            "Set OLLAMA_THINK=omit to skip the flag.",
            flush=True,
        )
        self.think = None
        return True

    @staticmethod
    def _parse(response, schema):
        content = strip_thinking(getattr(response.message, "content", "") or "")
        content = _FENCE.sub("", content).strip()
        if schema is None:
            return content
        if not content:
            raise ValueError(
                "The judge returned no JSON. Keep OLLAMA_THINK=false for "
                "models that think by default."
            )
        return schema.model_validate_json(content)
