"""
Rebuild (or append to) the ChromaDB vector store from data/Aventro Motors/.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --append
    python scripts/ingest.py --log-level DEBUG
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

from rag.config import CHROMA_COLLECTION_NAME, DEFAULT_LOG_LEVEL


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
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--append", action="store_true",
        help="Add to the existing collection instead of rebuilding it (default: rebuild).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=DEFAULT_LOG_LEVEL,
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def rebuild_collection(vector_store):
    """Delete the existing collection and recreate it empty."""
    logger = logging.getLogger(__name__)
    logger.info("Rebuilding collection: %s", CHROMA_COLLECTION_NAME)
    vector_store.client.delete_collection(name=CHROMA_COLLECTION_NAME)
    vector_store.collection = vector_store.client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "description": "PDF document embeddings for RAG",
            "hnsw:space": "cosine",
        },
    )
    logger.info("Collection reset. Count = %d", vector_store.collection.count())


def main():
    args = parse_args()

    _configure_logging(args.log_level)

    from rag.embedding.embedding_manager import EmbeddingManager
    from rag.retrieval.vector_store import VectorStore
    from rag.ingestion.loaders import process_all_documents
    from rag.ingestion.splitter import split_documents
    from rag.ingestion.indexer import index_documents

    logger = logging.getLogger(__name__)

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    if not args.append:
        rebuild_collection(vector_store)

    documents = process_all_documents()
    chunks = split_documents(documents)
    index_documents(chunks, embedding_manager, vector_store)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()