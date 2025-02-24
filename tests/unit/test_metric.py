"""Tests for the metrics."""

import math

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
