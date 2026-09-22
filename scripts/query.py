"""
Ask a question against the Aventro Motors RAG system.

Usage:
    python scripts/query.py "when was Aventro Motors founded"
    python scripts/query.py "What is ACC?" --top-k 5
    python scripts/query.py "What is ACC?" --mode advanced
    python scripts/query.py "What is ACC?" --mode flexible --top-k 3
    python scripts/query.py "What is ACC?" --log-level DEBUG
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Silence HuggingFace Hub BEFORE any HF-related imports
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv()

from rag.config import DEFAULT_SCORE_THRESHOLD, DEFAULT_LOG_LEVEL


def _configure_logging(level_name: str):
    """Configure logging: root=WARNING (suppress libs), rag=level, HF/httpx quiet."""
    import warnings

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )
    logging.getLogger("rag").setLevel(getattr(logging, level_name))
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Suppress HF Hub's "unauthenticated requests" warning (it's a warnings.warn,
    # not a logging call, so setLevel alone won't silence it)
    warnings.filterwarnings("ignore", module="huggingface_hub")
    warnings.filterwarnings("ignore", module="huggingface_hub.utils")


def parse_args():
    parser = argparse.ArgumentParser(description="Query the Aventro Motors RAG system.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Number of chunks to retrieve (default: from config).",
    )
    parser.add_argument(
        "--mode", choices=["simple", "flexible", "advanced"], default="simple",
        help="Which RAG pipeline to use (default: simple).",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help=f"Minimum similarity score (default: {DEFAULT_SCORE_THRESHOLD} from config).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def print_result(result: dict, show_sources: bool = True):
    print(result["answer"])
    if show_sources and result.get("sources"):
        print("\n=== Sources ===")
        for src in result["sources"]:
            print(f"  [score={src['score']:.4f}] {src['source']} (page {src['page']})")


def main():
    args = parse_args()

    _configure_logging(args.log_level)

    from rag.config import DEFAULT_TOP_K
    from rag.embedding.embedding_manager import EmbeddingManager
    from rag.retrieval.vector_store import VectorStore
    from rag.retrieval.retriever import RAGRetriever
    from rag.generation.llm import GroqLLM
    from rag.pipeline import rag_simple, rag_flexible, AdvancedRAGPipeline

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)
    groq_llm = GroqLLM()

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    min_score = (
        args.min_score if args.min_score is not None
        else DEFAULT_SCORE_THRESHOLD
    )

    print(f"\n--- Query (mode={args.mode}, top_k={top_k}, min_score={min_score}) ---")
    print(f"Q: {args.question}\n")

    if args.mode == "simple":
        result = rag_simple(
            args.question, rag_retriever, groq_llm,
            top_k=top_k, min_score=min_score,
        )
        print_result(result)
    elif args.mode == "flexible":
        result = rag_flexible(
            args.question, rag_retriever, groq_llm,
            top_k=top_k, min_score=min_score,
        )
        print_result(result)
    elif args.mode == "advanced":
        adv = AdvancedRAGPipeline(rag_retriever, groq_llm)
        result = adv.query(
            args.question,
            top_k=top_k,
            min_score=min_score,
            summarize=False,
        )
        print_result(result)


if __name__ == "__main__":
    main()