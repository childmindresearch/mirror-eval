"""Pydantic models for the outputs of LLMs."""

from pydantic import BaseModel, Field, validator
from typing import List


class BulletLengthError(ValueError):
    """Error for when a bullet point's length is invalid."""

class PersonaBulletError(ValueError):
    """Error for when a persona has incorrect number of bullets."""


class ConcerningAISummary(BaseModel):
    """Formatted output of the concerning summary call."""
    MIN_WORDS: int = 8
    MAX_WORDS: int = 12

    acknowledgement_1: str = Field(
        ..., description="First bullet acknowledging user's feelings"
    )
    acknowledgement_2: str = Field(
        ..., description="Second bullet acknowledging user's feelings"
    )
    positive_reframe: str = Field(
        ..., description="One bullet offering hope"
    )
    redirection: str = Field(
        ..., description="One bullet suggesting support resources"
    )

    @validator(
        "acknowledgement_1",
        "acknowledgement_2",
        "positive_reframe",
        "redirection",
    )
    def validate_length(cls, v: str) -> str:
        """Validate word count is within bounds."""
        words = len(v.split())
        if not cls.MIN_WORDS <= words <= cls.MAX_WORDS:
            raise BulletLengthError
        return v


class AIPersona(BaseModel):
    """Pydantic model for one AI persona."""
    name: str = Field(..., description="Name of the AI persona.")
    bullets: list[str] = Field(
        ...,
        description="Bullet points for the AI persona.",
        min_length=2,
        max_length=3,
    )


class AIPersonas(BaseModel):
    """Pydantic model for multiple AI personas."""
    personas: list[AIPersona] = Field(
        ...,
        description="List of AI personas.",
        min_length=2,
    )

    @validator("personas")
    def validate_length(cls, v: list[AIPersona]) -> list[AIPersona]:
        """Validate that each persona has the correct number of bullets."""
        for persona in v:
            if len(persona.bullets) < 2 or len(persona.bullets) > 3:
                raise PersonaBulletError
        return v

