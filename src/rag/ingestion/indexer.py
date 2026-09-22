"""
Indexing for the Aventro Motors RAG system.
"""
import logging
from typing import List

from langchain_core.documents import Document

from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def index_documents(
    chunks: List[Document],
    embedding_manager: EmbeddingManager,
    vector_store: VectorStore,
) -> None:
    """Embed and store document chunks into the vector store."""
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)