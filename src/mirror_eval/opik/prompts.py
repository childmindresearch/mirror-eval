"""Prompts for OPIK metrics."""

preference_prompt_single = """
Given the following initial prompt and two responses, determine which response is better

Initial prompt: {initial_prompt}

Response 1:
{first_response}

Response 2:
{second_response}

Return a JSON object with:
- "response": 1 if Response 1 is better, 2 if Response 2 is better
- "reason": A brief explanation of why you chose that response

Example:
{{
    "reason": "Response 2 provides more detailed and accurate information.",
    "response": 2
}}
"""

preference_prompt_double = """
Given the following two prompts and two responses, determine which response is better.
Both prompts represent the same task, but have slight differences in their instructions.
Determine which response is better given the instructions of the prompts.

First prompt: {first_prompt}

Second prompt: {second_prompt}

Response 1:
{first_response}

Response 2:
{second_response}

Return a JSON object with:
- "response": 1 if Response 1 is better, 2 if Response 2 is better
- "reason": A brief explanation of why you chose that response

Example:
{{
    "reason": "Response 2 provides more detailed and accurate information.",
    "response": 2
}}
"""
