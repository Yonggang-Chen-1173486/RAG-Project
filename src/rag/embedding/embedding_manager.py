"""
Embedding manager: converts text into vector embeddings.

Uses a local SentenceTransformer model (free, no API key needed).
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
        """
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded successfully. "
                f"Embedding dimension: {self.model.get_embedding_dimension()}"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings