"""
Indexing for the Aventro Motors RAG system.

Responsibility: given a list of document chunks, generate embeddings and
store them in a vector store. One function, one job.

It does NOT:
  - load files
  - split documents
  - initialize the LLM
"""
from typing import List

from langchain_core.documents import Document

from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore


def index_documents(
    chunks: List[Document],
    embedding_manager: EmbeddingManager,
    vector_store: VectorStore,
) -> None:
    """
    Embed and store document chunks into the vector store.

    Args:
        chunks: Documents to index (typically the output of split_documents).
        embedding_manager: Provides .generate_embeddings().
        vector_store: Provides .add_documents().
    """
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)