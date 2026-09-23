"""
Configuration for the Aventro Motors RAG system.

All tunable parameters are centralized here for easy experimentation.
"""
from pathlib import Path

# ============================================================
# Project paths
# ============================================================
# This file is at: <project_root>/src/rag/config.py
# So parents[2] gives us <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
AVENTRO_DATA_DIR = DATA_DIR / "Aventro Motors"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


# ============================================================
# Document loading
# ============================================================
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx",
    ".html", ".htm",
    ".md", ".txt", ".csv",
}


# ============================================================
# Text splitting
# ============================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# Embedding
# ============================================================
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768  # for gemini-embedding-001


# ============================================================
# Vector store (ChromaDB)
# ============================================================
CHROMA_COLLECTION_NAME = "pdf_documents"
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR)


# ============================================================
# Retrieval
# ============================================================
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.7


# ============================================================
# LLM (Groq)
# ============================================================
GROQ_MODEL_NAME = "Qwen/Qwen3.8-27B"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 800

# ============================================================
# Logging
# ============================================================
DEFAULT_LOG_LEVEL = "INFO"