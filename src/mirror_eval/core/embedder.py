from abc import ABC, abstractmethod
import ollama
from typing import Literal, Sequence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Embedder(ABC):
    """An abstract base class for embedders."""

    @abstractmethod
    def embed(self, text: str) -> Sequence[float]:
        """Embed a text."""

    @abstractmethod
    def get_similarity(self, text1: str, text2: str) -> float:
        """Get the similarity between two texts."""


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
