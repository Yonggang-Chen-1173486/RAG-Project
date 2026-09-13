"""
Text splitting for the Aventro Motors RAG system.

Responsibility: take a list of Documents and split each into smaller chunks,
returning a new list of Documents. It does NOT:
  - load files (that's loaders.py)
  - generate embeddings (that's embedding_manager.py)
  - store anything (that's indexer.py)
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents: Documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        A new list of Documents (each is a chunk).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs