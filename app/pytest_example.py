"""The same check, written the way DeepEval plugs into pytest.

Run it after Ollama is up and the model is pulled:

    deepeval test run app/pytest_example.py

``assert_test`` fails the test when the metric misses its threshold.
"""

from __future__ import annotations

from app.config import load_settings, prepare_process

prepare_process()

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from app.dataset import GOLDENS
from app.evaluate import build_metrics
from app.judge import LocalOllamaJudge
from app.ollama_client import ensure_model
from app.support_bot import answer


def test_return_policy_correctness():
    settings = load_settings()
    ensure_model(settings)
    judge = LocalOllamaJudge(settings)
    golden = GOLDENS[0]
    actual_output, retrieval_context = answer(settings, golden.question)
    test_case = LLMTestCase(
        input=golden.question,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_context,
        name=golden.name,
    )
    metric = build_metrics(settings, judge, "correctness")[0]
    assert_test(test_case=test_case, metrics=[metric], run_async=False)
