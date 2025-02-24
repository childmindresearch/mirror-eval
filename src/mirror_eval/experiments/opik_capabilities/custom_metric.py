"""Experiments used for testing new, custom metrics."""

from opik import evaluation

from mirror_eval.opik import metric, service


def run_custom_metric_experiment() -> None:
    """Experiment to test custom metrics in OPIK."""
    client = service.get_client()
    dataset = client.get_dataset("RAPI_test")
    query_metric = metric.QueryMetric(
        instruction="Evaluate whether the returned text is in French."
    )
    evaluation.evaluate_prompt(
        dataset=dataset,
        messages=[
            {
                "role": "user",
                "content": "Translate the following text to French: {{text}}",
            }
        ],
        model="gpt-3.5-turbo",
        scoring_metrics=[query_metric],
    )
