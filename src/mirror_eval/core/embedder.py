"""Module containing embedder implementations for text similarity."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity


class Embedder(ABC):
    """An abstract base class for embedders."""

    @abstractmethod
    def embed(self, text: str) -> Sequence[float]:
        """Embed a text."""

    @abstractmethod
    def get_similarity(self, text1: str, text2: str) -> float:
        """Get the similarity between two texts."""


class MockEmbedder(Embedder):
    """A mock embedder that returns predictable embeddings for testing."""

    def embed(self, text: str) -> Sequence[float]:
        """Return a predictable embedding based on the text."""
        if text == "Hello, world!":
            return [1.0, 0.0]  # First basis vector
        return [0.0, 1.0]  # Second basis vector

    def get_similarity(self, text1: str, text2: str) -> float:
        """Get cosine similarity between two texts using predictable embeddings."""
        embedding1 = np.array([self.embed(text1)])
        embedding2 = np.array([self.embed(text2)])
        return float(cosine_similarity(embedding1, embedding2)[0, 0])


class OllamaEmbedder(Embedder):
    """An embedder that uses Ollama."""

    def __init__(
        self,
        model: Literal[
            "nomic-embed-text",
            "mxbai-embed-large",
        ] = "mxbai-embed-large",
    ) -> None:
        """Initialize the embedding model."""
        self._model = model

    def embed(self, text: str) -> Sequence[float]:
        """Embed a text."""
        response = ollama.embed(model=self._model, input=text)
        return response.embeddings[0]

    def get_similarity(self, text1: str, text2: str) -> float:
        """Get the similarity between two texts using cosine similarity."""
        embedding1 = np.array([self.embed(text1)])  # Add batch dimension
        embedding2 = np.array([self.embed(text2)])
        return float(cosine_similarity(embedding1, embedding2)[0, 0])
