"""
Document loaders for the Aventro Motors RAG system.
"""
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document

from rag.config import AVENTRO_DATA_DIR

logger = logging.getLogger(__name__)


LOADER_BY_EXTENSION = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".md": TextLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

SUPPORTED_EXTENSIONS = set(LOADER_BY_EXTENSION)


def build_loader(file_path: Path):
    """Build the appropriate LangChain loader for a given file path."""
    extension = file_path.suffix.lower()
    if extension not in LOADER_BY_EXTENSION:
        raise ValueError(f"Unsupported extension: {extension}")

    loader_class = LOADER_BY_EXTENSION[extension]

    if loader_class is CSVLoader:
        return loader_class(file_path=str(file_path), encoding="utf-8")
    if loader_class is TextLoader:
        return loader_class(str(file_path), encoding="utf-8")
    if loader_class is BSHTMLLoader:
        return loader_class(str(file_path), open_encoding="utf-8")

    return loader_class(str(file_path))


def load_file(file_path: Path, root_directory: Path) -> List[Document]:
    """Load one file and enrich its metadata."""
    loader = build_loader(file_path)
    documents = loader.load()

    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    for doc in documents:
        doc.metadata.update({
            "source": str(file_path),
            "source_file": file_path.name,
            "source_path": str(file_path.relative_to(root_directory)),
            "file_type": file_path.suffix.lower().lstrip("."),
            "loader": type(loader).__name__,
        })

    return documents


def process_all_documents(data_directory=None) -> List[Document]:
    """Walk a directory, load every supported file, and return all Documents."""
    if data_directory is None:
        data_directory = AVENTRO_DATA_DIR

    data_dir = Path(data_directory)
    all_documents: List[Document] = []

    files = sorted(
        file_path
        for file_path in data_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Found %d supported files to process", len(files))

    for file_path in files:
        logger.debug("Processing: %s", file_path.relative_to(data_dir))
        try:
            documents = load_file(file_path, data_dir)
            all_documents.extend(documents)
            logger.debug("  Loaded %d document(s)", len(documents))
        except Exception as e:
            logger.error("  Error loading %s: %s", file_path, e)

    logger.info("Total documents loaded: %d", len(all_documents))
    return all_documents