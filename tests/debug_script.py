"""
Debug retrieval for a single query. only limit to 600 words of content.

Usage:
    python debug_q010.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import os
os.environ["HF_HUB_VERBOSITY"] = "error"

from dotenv import load_dotenv
load_dotenv()

from rag.embedding.embedding_manager import EmbeddingManager
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.retriever import RAGRetriever

embedding_manager = EmbeddingManager()
vector_store = VectorStore()
retriever = RAGRetriever(vector_store, embedding_manager)

query = "What is the boot space of the Aventro Grand SUV?"
results = retriever.retrieve(query, top_k=5, score_threshold=0.0)

print(f"\nRetrieved {len(results)} chunks for: '{query}'\n")
for i, r in enumerate(results, 1):
    print(f"--- Chunk {i} (score={r['similarity_score']:.4f}) ---")
    print(f"Source: {r['metadata'].get('source_file')}")
    print(f"Content: {r['content'][:600]}")
    print()