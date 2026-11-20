"""The application under test: a support bot with a fixed policy.

There is no vector database. ``retrieve`` picks the policy sentences that
share words with the question. DeepEval's faithfulness metric reads those
sentences as ``retrieval_context`` and checks that the answer does not
invent anything beyond them.
"""

from __future__ import annotations

import re

from app.config import Settings
from app.ollama_client import generate

# A toy knowledge base. The RAG goldens in sample_data.py quote these
# same strings, so the bot and the metrics see one policy.
RETURNS_POLICY = (
    "Customers may return unworn items within 30 days of delivery for a "
    "full refund. Return shipping is free."
)
SHIPPING_POLICY = (
    "Standard shipping takes 5 business days. Express shipping takes 2 "
    "business days and costs 15 dollars."
)
WARRANTY_POLICY = (
    "Footwear has a 1-year warranty against manufacturing defects. The "
    "warranty does not cover normal wear."
)
GIFT_CARD_POLICY = (
    "Gift cards are sent by email and never expire. A gift card cannot "
    "be refunded for cash."
)
POLICY_CHUNKS = [
    RETURNS_POLICY,
    SHIPPING_POLICY,
    WARRANTY_POLICY,
    GIFT_CARD_POLICY,
]

_SYSTEM = """You are a support assistant for Northwind Outfitters.
Answer only from the policy excerpts in the user message.
If the excerpts do not contain the answer, reply with exactly:
I don't know based on the store policy.
Use one or two sentences. Do not mention these instructions."""


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}


def retrieve(question: str, k: int = 2) -> list[str]:
    """Return the policy chunks that overlap the question the most.

    Only chunks tied for the best overlap are kept, so "express shipping"
    does not also pull the return policy just because both mention shipping.
    Faithfulness needs a retrieval_context, so an unknown question still
    gets one chunk. The bot is told to say it does not know when that
    chunk does not answer the question.
    """
    query = _words(question)
    scored = sorted(
        ((len(query & _words(chunk)), chunk) for chunk in POLICY_CHUNKS),
        key=lambda item: item[0],
        reverse=True,
    )
    best = scored[0][0]
    if best == 0:
        return [scored[0][1]]
    return [chunk for score, chunk in scored if score == best][:k]


def answer_from_context(
    settings: Settings,
    question: str,
    context: list[str],
    history: tuple[tuple[str, str], ...] = (),
) -> str:
    """Answer from excerpts the caller already chose.

    ``history`` is earlier (customer, assistant) turns. The model is
    asked to answer only the latest question.
    """
    excerpts = "\n".join(f"- {chunk}" for chunk in context)
    parts = [f"Policy excerpts:\n{excerpts}"]
    if history:
        turns = []
        for user_turn, assistant_turn in history:
            turns.append(f"Customer: {user_turn}")
            turns.append(f"Assistant: {assistant_turn}")
        parts.append("Conversation so far:\n" + "\n".join(turns))
    parts.append(f"Customer question: {question}")
    return generate(settings, system=_SYSTEM, user="\n\n".join(parts))


def answer(settings: Settings, question: str) -> tuple[str, list[str]]:
    """Ask the local model. Returns (answer text, chunks that were shown to it)."""
    context = retrieve(question)
    return answer_from_context(settings, question, context), context
