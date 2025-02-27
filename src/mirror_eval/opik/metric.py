"""Custom metrics for OPIK experiments."""

import json
import re
import statistics
from typing import Any

import opik
import pydantic
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import base_model, models_factory

from mirror_eval.core.embedder import (
    Embedder,
)
from mirror_eval.opik import prompts, template


class QueryResponse(pydantic.BaseModel):
    """The response model for the query metric."""

    response: str
    conclusion: bool


class PreferenceResponse(pydantic.BaseModel):
    """The response model for the preference metric."""

    response: int
    reason: str


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
        **_ignored_kwargs: dict[str, object],
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


class LlmStatementMetric(base_metric.BaseMetric):
    """A float metric based on True/False statements evaluated by an LLM."""

    def __init__(
        self,
        statements: list[str],
        model: str | base_model.OpikBaseModel | None = None,
        name: str = "Statement Model",
        *,
        track: bool = True,
    ) -> None:
        """Initialize the query metric.

        Args:
            statements: The statements to evaluate.
            model: The model to use in metric computation.
            name: The name of the metric.
            track: Whether to track the metric. Defaults to True.
        """
        super().__init__(name=name, track=track)
        self._statements = statements
        self._statements_tracked = False
        self._name = name

        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            self._model = models_factory.get(model_name=model)

    @opik.track()
    def _track_statements(self, _statements: list[str]) -> None:
        """Workaround to insert the statements into the tracking system.

        Args:
            _statements: The statements to track.
        """
        self._statements_tracked = True

    def score(
        self,
        input: str,  # noqa: A002
        output: str,
        **_ignored_kwargs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult:
        """Calculate score.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            **_ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            A ScoreResult with 1 if a match was found, 0 otherwise.
        """
        if not self._statements_tracked:
            self._track_statements(self._statements)

        model_output = self._model.generate_string(
            input=self._get_input(input, output),
            response_format=self._get_response_model(),
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
            A ScoreResult with 1 if a match was found, 0 otherwise.
        """
        if not self._statements_tracked:
            self._track_statements(self._statements)

        model_output = await self._model.agenerate_string(
            input=self._get_input(input, output),
            response_format=self._get_response_model(),
        )
        return self._parse_model_output(model_output)

    def _get_input(self, input: str, output: str) -> str:  # noqa: A002
        """Gets the input prompt for scoring.

        Returns:
            The input prompt.
        """
        statement_prompt = "- " + "\n- ".join(self._statements)
        return template.statements_base_prompt.format(
            input=input, output=output, statements=statement_prompt
        )

    def _get_response_model(self) -> object:
        """Creates a response model from the statements.

        This must be done dynamically to ensure that the 'statement'
        property is restricted to a copy of the input.

        Returns:
            The response model.
        """
        statement_models = []
        for statement in self._statements:
            statement_pattern = "^" + statement + "$"

            class Statement(pydantic.BaseModel):
                statement: str = pydantic.Field(..., pattern=statement_pattern)
                evaluation: str = pydantic.Field(..., min_length=20)
                conclusion: bool

            statement_models.append(Statement)

        class StatementModel(pydantic.BaseModel):
            statements: tuple[*statement_models]  # type: ignore[valid-type]

        return StatementModel

    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        """Extracts the scores from the LLM response.

        Returns:
            A ScoreResults wherein the value is the average score.
        """
        dict_content = json.loads(content)
        score = statistics.mean(response["conclusion"] for response in dict_content)
        return score_result.ScoreResult(
            name=self._name,
            value=score,
        )


class PreferenceMetric(base_metric.BaseMetric):
    """Metric that compares two responses and returns the better of the two."""

    def __init__(
        self,
        model: str | base_model.OpikBaseModel | None = None,
        name: str = "Preference Model",
        evaluation_instruction: str | None = None,
        *,
        track: bool = True,
    ) -> None:
        """Initialize the preference metric."""
        super().__init__(name=name, track=track)
        self._evaluation_instruction = (
            evaluation_instruction or prompts.preference_prompt_single
        )
        self._model = (
            model
            if isinstance(model, base_model.OpikBaseModel)
            else models_factory.get(model_name=model)
        )

    def score(
        self,
        initial_prompt: str,
        first_response: str,
        second_response: str,
        second_prompt: str | None = None,
        **_ignored_kwargs: Any,  # noqa: ANN401
    ) -> score_result.ScoreResult:
        """Return the better of the two responses.

        Args:
            initial_prompt: The initial prompt.
            first_response: The first response.
            second_response: The second response.
            second_prompt: The second prompt. [Optional]

        Returns:
            A ScoreResult with the better response and the reason.
        """
        if second_prompt is not None:
            self._evaluation_instruction = prompts.preference_prompt_double
            llm_query = self._evaluation_instruction.format(
                first_prompt=initial_prompt,
                second_prompt=second_prompt,
                first_response=first_response,
                second_response=second_response,
            )
        else:
            llm_query = self._evaluation_instruction.format(
                initial_prompt=initial_prompt,
                first_response=first_response,
                second_response=second_response,
            )
        model_output = json.loads(
            self._model.generate_string(
                input=llm_query,
                response_format=PreferenceResponse,
            )
        )
        return score_result.ScoreResult(
            name=self.name,
            value=model_output["response"],
            reason=model_output["reason"],
        )
