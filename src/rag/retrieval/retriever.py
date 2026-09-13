"""
Retriever for the Aventro Motors RAG system.

Responsibility:
  - accept a natural-language query
  - generate its embedding via EmbeddingManager
  - query the ChromaDB collection via VectorStore
  - return a ranked list of dicts (content + metadata + score)

It does NOT:
  - generate answers (that's generation/)
  - decide what to do with retrieved results (that's pipeline/)
"""
from typing import Any, Dict, List

from rag.config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings.
            embedding_manager: Manager for generating query embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query.
            top_k: Number of top results to return.
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of dicts: {id, content, metadata, similarity_score, distance, rank}.
            Empty list if no results or if an error occurs.
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

        # Process results
        retrieved_docs: List[Dict[str, Any]] = []

        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                # ChromaDB returns cosine *distance*; convert to similarity.
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

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        else:
            print("No documents found")

        return retrieved_docs