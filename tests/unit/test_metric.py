"""Tests for the metrics."""

import math
from collections.abc import Sequence
from typing import Any

import pytest
from opik.evaluation.models import base_model

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from mirror_eval.core.embedder import Embedder
from mirror_eval.opik import metric


class MockOpikModel(base_model.OpikBaseModel):
    """Directly mocking methods of opik models seems to fail for some reason.

    This just mocks the entire model instead.
    """

    def __init__(self) -> None:  # noqa: D107
        pass

    def generate_string(self, input: str, response_format: Any) -> str:  # type: ignore[override]  # noqa: A002, ANN401, ARG002, D102
        return '[{"conclusion": true}]'

    async def agenerate_string(self, input: str, response_format: Any) -> str:  # type: ignore[override]  # noqa: A002, ANN401, ARG002, D102
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
