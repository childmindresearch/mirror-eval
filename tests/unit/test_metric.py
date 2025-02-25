"""Tests for the metrics."""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from mirror_eval.core.embedder import Embedder
from mirror_eval.opik import metric


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
