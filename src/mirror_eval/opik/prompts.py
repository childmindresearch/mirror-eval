"""Prompts for OPIK metrics."""

preference_prompt = """
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
    "response": 2,
    "reason": "Response 2 provides more detailed and accurate information."
}}
"""
