"""Tests for the metrics."""

import math
from collections.abc import Sequence
from typing import Any

import litellm
import numpy as np
import pytest
import pytest_mock
from _pytest import logging as pytest_logging
from opik.evaluation.models import base_model
from sklearn.metrics.pairwise import cosine_similarity

from mirror_eval.core.embedder import Embedder
from mirror_eval.opik import metric


class MockOpikModel(base_model.OpikBaseModel):
    """Mock model for testing."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def generate_string(self, input: str, response_format: Any) -> str:  # type: ignore[override]  # noqa: A002, ANN401, ARG002, D102
        if response_format == metric.PreferenceResponse:
            return '{"response": 2, "reason": "Response 2 is more detailed"}'
        return '[{"conclusion": true}]'

    async def agenerate_string(self, input: str, response_format: Any) -> str:  # type: ignore[override]  # noqa: A002, ANN401, ARG002, D102
        if response_format == metric.PreferenceResponse:
            return '{"response": 2, "reason": "Response 2 is more detailed"}'
        return '[{"conclusion": true}]'

    def generate_provider_response(self) -> None:  # type: ignore[override]
        """Required for compliance with base model."""

    async def agenerate_provider_response(self) -> None:  # type: ignore[override]
        """Required for compliance with base model."""


def test_regex_metric_match() -> None:
    """Tests whether a matching regex returns a 1."""
    pattern = "^[0-9]$"
    string = "3"
    regex_metric = metric.RegexMetric(pattern)

    result = regex_metric.score(input="unused", output=string)

    assert math.isclose(result.value, 1)


def test_regex_metric_no_match() -> None:
    """Tests whether a matching regex returns a 1."""
    pattern = "^[0-9]$"
    string = "a"
    regex_metric = metric.RegexMetric(pattern)

    result = regex_metric.score(input="unused", output=string)

    assert math.isclose(result.value, 0)


class MockEmbedder(Embedder):
    """A mock embedder that returns predictable embeddings for testing."""

    def embed(self, text: str) -> Sequence[float]:
        """Return a predictable embedding based on the text."""
        if text == "Hello, world!":
            return [1.0, 0.0]  # First basis vector
        return [0.0, 1.0]  # Second basis vector

    def get_similarity(self, text1: str, text2: str) -> float:
        """Get cosine similarity between two texts using predictable embeddings."""
        embedding1 = np.array([self.embed(text1)])
        embedding2 = np.array([self.embed(text2)])
        return float(cosine_similarity(embedding1, embedding2)[0, 0])


@pytest.fixture
def mock_embedder() -> Embedder:
    """Fixture providing a mock embedder for testing."""
    return MockEmbedder()


@pytest.fixture
def embedding_metric(mock_embedder: Embedder) -> metric.EmbeddingMetric:
    """Fixture for the embedding metric."""
    return metric.EmbeddingMetric(mock_embedder)


def test_embedding_metric_score(embedding_metric: metric.EmbeddingMetric) -> None:
    """Test the embedding metric score."""
    result = embedding_metric.score(
        first_response="Hello, world!",
        second_response="Hello, world!",
    )
    assert math.isclose(result.value, 1.0)


def test_statement_metric() -> None:
    """Tests the statement metric happy path."""
    statements = ["This text is in French."]
    input_text = "An input text."
    output_text = "Oui, ce texte est en français."
    statement_metric = metric.LlmStatementMetric(statements, MockOpikModel())

    result = statement_metric.score(input=input_text, output=output_text)

    assert result.name == "Statement Model"
    assert result.value == 1


def test_statement_metric_fallback_not_strict(
    caplog: pytest_logging.LogCaptureFixture, mocker: pytest_mock.MockerFixture
) -> None:
    """Tests the fallback for non-strict models works."""
    call_count = 0

    def side_effect(*_args: Any, **_kwargs: Any) -> str:  # noqa: ANN401
        """Raise an error only on the first (strict) attempt."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "Invalid schema for response_format ..."
            raise litellm.BadRequestError(msg, "", "")
        return '[{"conclusion": true}]'

    spy = mocker.patch.object(
        MockOpikModel, "agenerate_string", side_effect=side_effect
    )
    statements = ["This text is in French."]
    input_text = "An input text."
    output_text = "Oui, ce texte est en français."
    statement_metric = metric.LlmStatementMetric(statements, MockOpikModel())

    result = statement_metric.score(input=input_text, output=output_text)

    assert result.name == "Statement Model"
    assert result.value == 1
    assert spy.call_count == 2  # noqa: PLR2004
    assert "Could not run this model with strict properties." in caplog.text


@pytest.mark.asyncio
async def test_async_statement_metric() -> None:
    """Tests the statement metric async happy path."""
    statements = ["This text is in French."]
    input_text = "An input text."
    output_text = "Oui, ce texte est en français."
    statement_metric = metric.LlmStatementMetric(statements, MockOpikModel())

    result = await statement_metric.ascore(input=input_text, output=output_text)

    assert result.name == "Statement Model"
    assert result.value == 1


def test_preference_metric() -> None:
    """Tests the preference metric."""
    initial_prompt = "Explain quantum computing."
    first_response = "Quantum computers use qubits."
    second_response = "A detailed explanation of quantum superposition..."
    preference_metric = metric.PreferenceMetric(MockOpikModel())

    result = preference_metric.score(
        initial_prompt=initial_prompt,
        first_response=first_response,
        second_response=second_response,
    )

    assert result.name == "Preference Model"
    assert "more detailed" in result.reason.lower()


def test_logprobs_get_final_score() -> None:
    """Test the logprobs scoring function directly."""
    metric_instance = metric.LogprobsMetric(model="gpt-4o-mini")
    mock_logprobs = [
        type("Logprob", (), {"token": "7", "logprob": -0.1}),
        type("Logprob", (), {"token": "3", "logprob": -1.0}),
        type("Logprob", (), {"token": "{", "logprob": -5.0}),
    ]

    result = metric_instance._get_final_score(mock_logprobs)  # noqa: SLF001

    total_probs = [np.exp(-0.1), np.exp(-1.0)]
    normalized_probs = [prob / sum(total_probs) for prob in total_probs]
    value = 7 * normalized_probs[0] + 3 * normalized_probs[1]
    assert math.isclose(result, value)
