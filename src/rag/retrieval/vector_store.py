"""
ChromaDB-backed vector store for the Aventro Motors RAG system.
"""
import logging
import os
import uuid
from typing import Any, List

import chromadb
import numpy as np

from rag.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG",
                    "hnsw:space": "cosine",
                },
            )
            logger.info("Vector store initialized. Collection: %s", self.collection_name)
            logger.info("Existing documents in collection: %d", self.collection.count())

        except Exception as e:
            logger.error("Error initializing vector store: %s", e)
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        logger.info("Adding %d documents to vector store...", len(documents))

        ids: List[str] = []
        metadatas: List[dict] = []
        documents_text: List[str] = []
        embeddings_list: List[list] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            logger.info("Successfully added %d documents to vector store", len(documents))
            logger.info("Total documents in collection: %d", self.collection.count())

        except Exception as e:
            logger.error("Error adding documents to vector store: %s", e)
            raise