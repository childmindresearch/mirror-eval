from mirror_eval.opik import service
from opik import evaluation
from mirror_eval.opik import metrics


def run_custom_metric_experiment() -> None:
    client = service.get_client()
    dataset = client.get_dataset("RAPI_test")
    result = evaluation.evaluate_prompt(
        dataset=dataset,
        messages=[
            {
                "role": "user",
                "content": "Translate the following text to French: {{input}}",
            }
        ],
        model="gpt-3.5-turbo",  # or your preferred model
        scoring_metrics=[metrics.CustomMetric()],
    )
