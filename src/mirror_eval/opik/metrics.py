import random
from typing import Any

from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import base_model, models_factory


class CustomMetric(base_metric.BaseMetric):
    """Retained solely as an example for building future metrics.

    Delete this once a proper metric has been built.
    """

    def __init__(
        self,
        model: str | base_model.OpikBaseModel | None = None,
        name: str = "Structured Output",
        track: bool = True,
    ):
        self.name = name
        self.track = track
        if isinstance(model, base_model.OpikBaseModel):
            self._model = model
        else:
            self._model = models_factory.get(model_name=model)

    def score(
        self, input: str, output: str, **ignored_kwargs: Any
    ) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Calculate the score for the given input and output.

        Args:
            input: The original input/question.
            output: The LLM's output to evaluate.
            **ignored_kwargs: Additional keyword arguments that are ignored.

        Returns:
            score_result.ScoreResult: A ScoreResult object with a random value.
        """
        return score_result.ScoreResult(
            name=self.name,
            value=random.random(),
        )
