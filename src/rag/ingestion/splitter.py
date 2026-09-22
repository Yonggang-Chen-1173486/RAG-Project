"""
Text splitting for the Aventro Motors RAG system.
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split a list of Documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(split_docs))

    if split_docs:
        logger.debug("Example chunk content: %s...", split_docs[0].page_content[:200])
        logger.debug("Example chunk metadata: %s", split_docs[0].metadata)

    return split_docs