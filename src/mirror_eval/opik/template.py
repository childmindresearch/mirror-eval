"""Prompt templating for OPIK metrics."""

query_base_prompt = """You will be given an instruction by the user.
Your task is to respond to this instruction, and return a boolean conclusion.
The meaning of True/False in this boolean should be clear from the user's
instructions.

USER INSTRUCTION:
{instruction}

INPUT
{input}

OUTPUT
{output}

It is crucial that you provide your answer in the following JSON format:
{{
    "response": "A response to this instruction."
    "conclusion": true
}}
"""
