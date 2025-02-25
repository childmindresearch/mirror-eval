"""Tests for the metrics."""

import math

import pytest

from mirror_eval.core.embedder import (
    Embedder,
    MockEmbedder,
)
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


@pytest.fixture
def embedder() -> Embedder:
    """Fixture for the embedder."""
    return MockEmbedder()


@pytest.fixture
def embedding_metric(embedder: Embedder) -> metric.EmbeddingMetric:
    """Fixture for the embedding metric."""
    return metric.EmbeddingMetric(embedder)


def test_embedding_metric_score(embedding_metric: metric.EmbeddingMetric) -> None:
    """Test the embedding metric score."""
    result = embedding_metric.score(
        first_response="Hello, world!",
        second_response="Hello, world!",
    )
    assert math.isclose(result.value, 1.0)
