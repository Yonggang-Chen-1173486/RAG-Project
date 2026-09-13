"""
ChromaDB-backed vector store for the Aventro Motors RAG system.

Responsibility:
  - manage a persistent ChromaDB client + collection
  - provide an add_documents(documents, embeddings) method
  - expose the underlying collection for querying

It does NOT:
  - generate embeddings (that's embedding_manager.py)
  - decide when to index (that's indexer.py)
  - run retrieval (that's retriever.py)
"""
import os
from typing import Any, List

import chromadb
import numpy as np

from rag.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


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
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain Documents.
            embeddings: Corresponding embeddings (numpy array, shape N x D).
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(f"Adding {len(documents)} documents to vector store...")

        import uuid  # local import — only needed here

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
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise