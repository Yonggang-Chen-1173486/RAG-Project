"""
Retriever for the Aventro Motors RAG system.
"""
import logging
from typing import Any, Dict, List

from rag.config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        logger.info("Retrieving documents for query: '%s'", query)
        logger.debug("Top K: %d, Score threshold: %s", top_k, score_threshold)

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            return []

        retrieved_docs: List[Dict[str, Any]] = []

        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance,
                        "rank": i + 1,
                    })

            logger.info("Retrieved %d documents (after filtering)", len(retrieved_docs))
        else:
            logger.warning("No documents found")

        return retrieved_docs