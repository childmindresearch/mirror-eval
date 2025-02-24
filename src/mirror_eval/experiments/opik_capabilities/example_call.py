"""Example call to Opik."""

import opik
from opik.evaluation import evaluate
from mirror_eval.opik import service
from mirror_eval.models.outputs import ConcerningAISummary
from mirror_eval.opik.metric import QueryMetric
from opik import track
import ollama
import json

from typing import Any

from loguru import logger

DATASET_NAME = "RAPI_test"
PROMPT_NAME = "Concerning AI Summary 2/20"
MODEL = "llama3.2:3b"

client = service.get_client()
dataset = client.get_dataset(DATASET_NAME)
prompt = dataset.get_prompt(PROMPT_NAME)

@track()
def call_llm(dataset_item: dict[str, Any]) -> dict[str, Any]:
    """An actual call to the LLM."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt.prompt
            },
            {
                "role": "user",
                "content": dataset_item["input"],
            }
        ],
        format=ConcerningAISummary.model_json_schema(),
    )
    return json.loads(response.message.content)

@track()
def evaluation_task(dataset_item: dict[str, Any]) -> dict[str, Any]:
    """A runnable evaluation task."""
    return {
        "output": call_llm(dataset_item),
    }


def run() -> None:
    """Run the example call to Opik."""
    query_metric = QueryMetric(
        instruction="Evaluate whether the returned text is in French."
    )
    evaluate(
        dataset=dataset,
        prompt=prompt,
        task=evaluation_task,
        experiment_name="Test experiment",
        nb_samples=10,
        scoring_metrics=[query_metric],
    )


if __name__ == "__main__":
    run()