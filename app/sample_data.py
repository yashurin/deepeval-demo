"""Sample questions for the local evaluation run.

Eight cases are RAG-style. Each one carries the policy excerpt the bot
is allowed to see, and that same excerpt is the retrieval context
DeepEval scores. One of those is a follow-up turn: the earlier exchange
is stored on ``history``. Three cases are plain questions with no document.

The Antarctica case is shown an excerpt that does not mention shipping
destinations. The expected answer is the bot's abstention sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.support_bot import (
    GIFT_CARD_POLICY,
    RETURNS_POLICY,
    SHIPPING_POLICY,
    WARRANTY_POLICY,
)


@dataclass(frozen=True)
class Golden:
    name: str
    question: str
    expected_output: str
    # None marks a plain question. A tuple marks a RAG case: the bot
    # answers from these excerpts, and the metrics see the same text.
    retrieval_context: tuple[str, ...] | None = None
    # Earlier (customer, assistant) turns. The model answers `question`
    # with this conversation in the prompt. Single-turn metrics then see
    # the whole transcript as the test input.
    history: tuple[tuple[str, str], ...] = ()


def case_input(golden: Golden) -> str:
    """The text DeepEval stores as ``input``.

    A follow-up includes the earlier turns so relevancy and correctness
    can see what the new question is about.
    """
    if not golden.history:
        return golden.question
    lines: list[str] = []
    for user_turn, assistant_turn in golden.history:
        lines.append(f"Customer: {user_turn}")
        lines.append(f"Assistant: {assistant_turn}")
    lines.append(f"Customer: {golden.question}")
    return "\n".join(lines)


GOLDENS = [
    Golden(
        name="returns",
        question="What is your return policy if the shoes don't fit?",
        expected_output=(
            "Unworn items can be returned within 30 days of delivery for a "
            "full refund, and return shipping is free."
        ),
        retrieval_context=(RETURNS_POLICY,),
    ),
    Golden(
        name="express-shipping",
        question="How long does express shipping take, and what does it cost?",
        expected_output="Express shipping takes 2 business days and costs 15 dollars.",
        retrieval_context=(SHIPPING_POLICY,),
    ),
    Golden(
        name="standard-shipping",
        question="How long does standard shipping take?",
        expected_output="Standard shipping takes 5 business days.",
        retrieval_context=(SHIPPING_POLICY,),
    ),
    Golden(
        name="warranty",
        question="What warranty do shoes have, and does it cover worn-out soles?",
        expected_output=(
            "Footwear has a 1-year warranty against manufacturing defects. "
            "Normal wear is not covered."
        ),
        retrieval_context=(WARRANTY_POLICY,),
    ),
    Golden(
        name="unknown-destination",
        question="Do you ship to Antarctica?",
        expected_output="I don't know based on the store policy.",
        # The excerpt is real policy and irrelevant to the question.
        # Contextual relevancy should reject it. The abstention itself
        # adds no claim, so faithfulness can still pass.
        retrieval_context=(RETURNS_POLICY,),
    ),
    Golden(
        name="gift-card",
        question="Do gift cards expire, and can I cash one out?",
        expected_output=(
            "Gift cards never expire. They cannot be refunded for cash."
        ),
        retrieval_context=(GIFT_CARD_POLICY,),
    ),
    Golden(
        name="express-and-returns",
        question=(
            "How long does express shipping take, and how many days do I "
            "have to return unworn shoes?"
        ),
        expected_output=(
            "Express shipping takes 2 business days. Unworn items can be "
            "returned within 30 days of delivery."
        ),
        retrieval_context=(SHIPPING_POLICY, RETURNS_POLICY),
    ),
    Golden(
        name="return-window-followup",
        question="What if I already wore them once?",
        expected_output=(
            "Worn items are not eligible for return. The 30-day return "
            "is for unworn items."
        ),
        retrieval_context=(RETURNS_POLICY,),
        history=(
            (
                "What is your return policy if the shoes don't fit?",
                "Unworn items can be returned within 30 days of delivery "
                "for a full refund, and return shipping is free.",
            ),
        ),
    ),
    Golden(
        name="capital-of-france",
        question="What is the capital of France?",
        expected_output="The capital of France is Paris.",
    ),
    Golden(
        name="days-in-a-week",
        question="How many days are in a week?",
        expected_output="There are 7 days in a week.",
    ),
    Golden(
        name="six-times-seven",
        question="What is 6 times 7?",
        expected_output="6 times 7 is 42.",
    ),
]
