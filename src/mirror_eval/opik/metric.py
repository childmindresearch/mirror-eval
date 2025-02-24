"""Custom metrics for OPIK experiments."""

import json
import re
from typing import Any

import opik
import pydantic
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import base_model, models_factory

from mirror_eval.opik import template
from mirror_eval.core.embedder import (
    Embedder,
    OllamaEmbedder,
)


class QueryResponse(pydantic.BaseModel):
    """The response model for the query metric."""

    response: str
    conclusion: bool


class QueryMetric(base_metric.BaseMetric):
    """A boolean metric on whether the output adheres to a custom instruction."""

    def __init__(
        self,
        instruction: str,
        model: str | base_model.OpikBaseModel | None = None,
        name: str = "Query Model",
        *,
        track: bool = True,
    ) -> None:
        """Initialize the query metric.

        Args:
            instruction: The instruction to evaluate.
            model: The model to use in metric computation.
            name: The name of the metric.
            track: Whether to track the metric. Defaults to True.
        """
        super().__init__(name=name, track=track)
        self._instruction = instruction
        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            self._model = models_factory.get(model_name=model)
        self._instruction_tracked = False

    @opik.track()
    def _track_instruction(self, _instruction: str) -> None:
        """Workaround to insert the instruction into the tracking system.

        Args:
            _instruction: The instructions to track.
        """
        self._instruction_tracked = True

    def score(
        self,
        input: str,  # noqa: A002
        output: str,
        **_ignored_kwargs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult:
        """Calculate the score for the given input and output.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            **_ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            A ScoreResult object, 1 if the instruction returns True, 0 otherwise.
        """
        if not self._instruction_tracked:
            self._track_instruction(self._instruction)

        llm_query = template.query_base_prompt.format(
            instruction=self._instruction, input=input, output=output
        )
        model_output = self._model.generate_string(
            input=llm_query, response_format=QueryResponse
        )
        return self._parse_model_output(model_output)

    async def ascore(
        self,
        input: str,  # noqa: A002
        output: str,
        **_ignored_kwargs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult:
        """Calculate the score for the given input and output.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            **_ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            A ScoreResult object, 1 if the instruction returns True, 0 otherwise.
        """
        if not self._instruction_tracked:
            self._track_instruction(self._instruction)

        llm_query = template.query_base_prompt.format(
            instruction=self._instruction, input=input, output=output
        )
        model_output = await self._model.agenerate_string(
            input=llm_query, response_format=QueryResponse
        )
        return self._parse_model_output(model_output)

    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        dict_content = json.loads(content)
        score = float(dict_content["conclusion"])
        return score_result.ScoreResult(
            name=self.name,
            value=float(score),
        )


class RegexMetric(base_metric.BaseMetric):
    """A boolean metric on whether the output adheres to a regular expression."""

    def __init__(
        self,
        regex: str,
        name: str = "Regular Expression",
        *,
        track: bool = True,
    ) -> None:
        """Initialize the query metric.

        Args:
            regex: The regular expression to evaluate.
            name: The name of the metric.
            track: Whether to track the metric. Defaults to True.
        """
        super().__init__(name=name, track=track)
        self._pattern = re.compile(regex)
        self._pattern_tracked = False

    @opik.track()
    def _track_instruction(self, _regex: str) -> None:
        """Workaround to insert the regex into the tracking system.

        Args:
            _regex: The instructions to track.
        """
        self._pattern_tracked = True

    def score(
        self,
        input: str,  # noqa: A002, ARG002
        output: str,
        **_ignored_kwargs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult:
        """Calculate the score for the given input and output.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            **_ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            A ScoreResult with 1 if a match was found, 0 otherwise.
        """
        if not self._pattern_tracked:
            self._track_instruction(self._pattern.pattern)

        match = self._pattern.match(output)
        return score_result.ScoreResult(
            name=self.name,
            value=float(bool(match)),
        )


class EmbeddingMetric(base_metric.BaseMetric):
    """Class to evaluate the similarity between two outputs."""

    def __init__(
        self,
        embedder: Embedder,
        name: str = "Embedding Metric",
        *,
        track: bool = True,
    ) -> None:
        """Initialize the embedding metric."""
        super().__init__(name=name, track=track)
        self._embedder = embedder

    def score(
        self,
        first_response: str,
        second_response: str,
        **_ignored_kwargs: Any,
    ) -> score_result.ScoreResult:
        """Calculate the score for the given input and output.

        Args:
            first_response: The first response to evaluate.
            second_response: The second response to evaluate.
        """
        similarity = self._embedder.get_similarity(
            text1=first_response, text2=second_response
        )
        return score_result.ScoreResult(
            name=self.name,
            value=similarity,
        )
