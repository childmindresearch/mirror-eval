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

statements_base_prompt = """You will be given an input, an output, and
a set of statements. Your task is to verify whether the statements are True or False
based on the output.

<input>
{input}
</input>

<output>
{output}
</output

<statements>
{statements}
</statements>

Return your response as a JSON object as follows:

{{
    "statement": "The statement",
    "evaluation": "An evaluation paragraph of the statement.",
    "conclusion": true/false
}}

The statement should hold a copy of the statement that will be evaluated.

"""
