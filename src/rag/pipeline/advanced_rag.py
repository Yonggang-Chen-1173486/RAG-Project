"""
Advanced RAG pipeline for the Aventro Motors RAG system.

Features (all in one class, for convenience):
  - retrieval + generation
  - source citations with preview
  - confidence score (top similarity)
  - optional "streaming" simulation (prints the prompt slowly)
  - optional answer summarization (a second LLM call)
  - query history

It does NOT:
  - initialize the retriever or the LLM (the caller does that)
  - persist history to disk (in-memory only)
"""
import time
from typing import Any, Dict, List
from rag.config import DEFAULT_SCORE_THRESHOLD
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM


class AdvancedRAGPipeline:
    """RAG pipeline with citations, confidence, history, and optional extras."""

    def __init__(self, retriever: RAGRetriever, llm: GroqLLM):
        self.retriever = retriever
        self.llm = llm
        self.history: List[Dict[str, Any]] = []  # query history

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = DEFAULT_SCORE_THRESHOLD,
        stream: bool = False,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a full RAG query with optional streaming and summarization.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            min_score: Minimum similarity score to keep a chunk.
            stream: If True, print the prompt slowly (visual simulation only).
            summarize: If True, also ask the LLM for a 2-sentence summary.

        Returns:
            Dict with: question, answer (with citations), sources, summary, history.
        """
        # 1. Retrieve
        results = self.retriever.retrieve(
            question, top_k=top_k, score_threshold=min_score
        )

        if not results:
            answer = "No relevant context found."
            sources: List[Dict[str, Any]] = []
            context = ""
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [
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

            # 2. Optional "streaming" simulation (prints the prompt, not the answer)
            prompt = (
                f"Use the following context to answer the question concisely.\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i + 80], end="", flush=True)
                    time.sleep(0.05)
                print()

            # 3. Generate
            response = self.llm.invoke([prompt])
            answer = response.content

        # 4. Build citations
        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations)
            if citations
            else answer
        )

        # 5. Optional summarization
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # 6. Store history
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
        })

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }