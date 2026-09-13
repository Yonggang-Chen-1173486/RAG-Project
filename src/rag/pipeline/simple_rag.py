"""
Simple RAG pipeline for the Aventro Motors RAG system.

Behavior:
  - if the retrieved chunks pass the score threshold -> answer from context
  - otherwise -> return a fixed message saying no relevant info was found
"""
from rag.config import DEFAULT_SCORE_THRESHOLD
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM

NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the Aventro Motors "
    "documents to answer this question."
)


def rag_simple(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = 3,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """
    Retrieve context and generate an answer.

    Args:
        query: User question.
        retriever: RAGRetriever instance.
        llm: GroqLLM instance.
        top_k: Number of chunks to retrieve.
        min_score: Minimum similarity score for a chunk to be considered relevant.

    Returns:
        Answer string, or NO_CONTEXT_MESSAGE if no relevant chunks were found.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return NO_CONTEXT_MESSAGE

    context = "\n\n".join([doc["content"] for doc in results])
    prompt = f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""

    response = llm.llm.invoke([prompt])
    return response.content