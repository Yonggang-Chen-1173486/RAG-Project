"""
Rebuild (or append to) the ChromaDB vector store from data/Aventro Motors/.

Usage:
    python scripts/ingest.py              # default: rebuild from scratch
    python scripts/ingest.py --append     # add to existing collection
"""
import argparse
import sys
from pathlib import Path

# Make `rag` importable when running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import CHROMA_COLLECTION_NAME
from rag.embedding.embedding_manager import EmbeddingManager
from rag.ingestion.loaders import process_all_documents
from rag.ingestion.splitter import split_documents
from rag.ingestion.indexer import index_documents
from rag.retrieval.vector_store import VectorStore


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing collection instead of rebuilding it (default: rebuild).",
    )
    return parser.parse_args()


def rebuild_collection(vector_store: VectorStore):
    """Delete the existing collection and recreate it empty."""
    print(f"Rebuilding collection: {CHROMA_COLLECTION_NAME}")
    vector_store.client.delete_collection(name=CHROMA_COLLECTION_NAME)
    # Recreate it
    vector_store.collection = vector_store.client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "description": "PDF document embeddings for RAG",
            "hnsw:space": "cosine",
        },
    )
    print(f"Collection reset. Count = {vector_store.collection.count()}")


def main():
    args = parse_args()

    # 1. Init components
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    # 2. Optionally rebuild
    if not args.append:
        rebuild_collection(vector_store)

    # 3. Load → split → index
    documents = process_all_documents()
    chunks = split_documents(documents)
    index_documents(chunks, embedding_manager, vector_store)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()