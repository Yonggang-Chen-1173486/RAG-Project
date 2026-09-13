"""
Flexible RAG pipeline for the Aventro Motors RAG system.

Behavior:
  - if relevant chunks were retrieved -> answer from context
  - otherwise -> state that no relevant info was found, then answer from
    the LLM's general knowledge
"""
from rag.config import DEFAULT_SCORE_THRESHOLD
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM


def rag_flexible(
    query: str,
    retriever: RAGRetriever,
    llm: GroqLLM,
    top_k: int = 3,
    min_score: float = DEFAULT_SCORE_THRESHOLD,
) -> str:
    """
    Retrieve context and generate an answer; fall back to general knowledge
    if no relevant context is found.

    Args:
        query: User question.
        retriever: RAGRetriever instance.
        llm: GroqLLM instance.
        top_k: Number of chunks to retrieve.
        min_score: Minimum similarity score for a chunk to be considered relevant.

    Returns:
        Answer string.
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

    response = llm.llm.invoke([prompt])
    return response.content