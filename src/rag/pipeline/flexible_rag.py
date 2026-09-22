"""
Flexible RAG pipeline for the Aventro Motors RAG system.

Behavior:
  - if relevant chunks were retrieved -> answer from context
  - otherwise -> state that no relevant info was found, then answer from
    the LLM's general knowledge

Returns:
  A dict with keys: "answer" (str) and "sources" (list of dicts).
"""
from typing import Any, Dict, List

from rag.config import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM


def _build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert retriever results into a sources list (source, score, preview)."""
    return [
        {
            "source": doc["metadata"].get(
                "source_file", doc["metadata"].get("source", "unknown")
            ),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:120] + "...",
        }
        for doc in results
    ]


def rag_flexible(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Retrieve context and generate an answer; fall back to general knowledge
    if no relevant context is found.

    Returns:
        {
          "answer": str,
          "sources": list of {"source", "page", "score", "preview"}
        }
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if results:
        context = "\n\n".join([doc["content"] for doc in results])
        prompt = f"""Use the following context to answer the question.
        If the context doesn't contain the answer, say so.

        Context: {context}
        Question: {query}
        Answer:"""
    else:
        prompt = f"""The Aventro Motors knowledge base does not contain relevant
        information to answer this question.

        Please do two things in your answer:
        1. Start by stating clearly that no relevant information was found
           in the Aventro Motors documents.
        2. Then answer the question using your general knowledge.
           If you're not sure, say so.

        Question: {query}
        Answer:"""

    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": _build_sources(results) if results else [],
    }