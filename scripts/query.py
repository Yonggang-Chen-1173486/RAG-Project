"""
Ask a question against the Aventro Motors RAG system.

Usage:
    python scripts/query.py "when was Aventro Motors founded"
    python scripts/query.py "What is ACC?" --top-k 5
    python scripts/query.py "What is ACC?" --mode advanced
    python scripts/query.py "What is ACC?" --mode flexible --top-k 3
    python scripts/query.py "What is ACC?" --min-score 0.2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from rag.config import DEFAULT_SCORE_THRESHOLD
from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm import GroqLLM
from rag.pipeline import rag_simple, rag_flexible, AdvancedRAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Query the Aventro Motors RAG system.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--mode", choices=["simple", "flexible", "advanced"], default="simple",
        help="Which pipeline to use (default: simple).",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help=f"Minimum similarity score (default: {DEFAULT_SCORE_THRESHOLD} from config).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    groq_llm = GroqLLM()

    # Resolve min_score: explicit flag > config default
    min_score = (
        args.min_score if args.min_score is not None
        else DEFAULT_SCORE_THRESHOLD
    )

    print(f"\n--- Query (mode={args.mode}, top_k={args.top_k}, min_score={min_score}) ---")
    print(f"Q: {args.question}\n")

    if args.mode == "simple":
        answer = rag_simple(
            args.question, rag_retriever, groq_llm,
            top_k=args.top_k, min_score=min_score,
        )
        print(answer)

    elif args.mode == "flexible":
        answer = rag_flexible(
            args.question, rag_retriever, groq_llm,
            top_k=args.top_k, min_score=min_score,
        )
        print(answer)

    elif args.mode == "advanced":
        adv = AdvancedRAGPipeline(rag_retriever, groq_llm.llm)
        result = adv.query(
            args.question,
            top_k=args.top_k,
            min_score=min_score,
            stream=False,
            summarize=False,
        )
        print(result["answer"])


if __name__ == "__main__":
    main()