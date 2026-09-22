"""
Simple RAG pipeline for the Aventro Motors RAG system.

Behavior:
  - if the retrieved chunks pass the score threshold -> answer from context
  - otherwise -> return a fixed message saying no relevant info was found

Returns:
  A dict with keys: "answer" (str) and "sources" (list of dicts).
"""
from typing import Any, Dict, List

from rag.config import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM

NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the Aventro Motors "
    "documents to answer this question."
)


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


def rag_simple(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Retrieve context and generate an answer.

    Returns:
        {
          "answer": str,
          "sources": list of {"source", "page", "score", "preview"}
        }
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {"answer": NO_CONTEXT_MESSAGE, "sources": []}

    context = "\n\n".join([doc["content"] for doc in results])
    prompt = f"""Use ONLY the following context to answer the question.
        If the context does NOT contain enough information to answer the question,
        respond with exactly: "I couldn't find relevant information in the Aventro Motors documents to answer this question."
        Do NOT use your general knowledge.

        Context:
        {context}

        Question: {query}

        Answer:"""

    answer = llm.invoke(prompt)
    return {"answer": answer, "sources": _build_sources(results)}