"""
Embedding generation for the Aventro Motors RAG system.

Uses Google Generative AI (Gemini) for embeddings via LangChain integration.
"""
import logging
from typing import List

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Handles document embedding generation using Google Gemini."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.embeddings = None
        self._load_model()

    def _load_model(self):
        """Initialize the Google Generative AI embeddings client."""
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            # 初始化 LangChain 的 Google Embeddings 客户端
            # 会自动从环境变量 GOOGLE_API_KEY 读取密钥
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                output_dimensionality=EMBEDDING_DIMENSION,  # 指定输出维度
            )
            # 做个简单的测试调用，确保能连上 API
            test_vector = self.embeddings.embed_query("test")
            logger.info(
                "Model loaded successfully. Embedding dimension: %d",
                len(test_vector)
            )
        except Exception as e:
            logger.error("Error loading model %s: %s", self.model_name, e)
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.embeddings:
            raise ValueError("Model not loaded")

        logger.info("Generating embeddings for %d texts...", len(texts))
        # LangChain 的 embed_documents 会处理批量请求
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings = np.array(embeddings_list)
        
        logger.info("Generated embeddings with shape: %s", embeddings.shape)
        return embeddings