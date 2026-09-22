"""Document ingestion module: loading, splitting, indexing."""

from .loaders import process_all_documents, load_file, build_loader
from .splitter import split_documents
from .indexer import index_documents

__all__ = [
    "process_all_documents",
    "load_file",
    "build_loader",
    "split_documents",
    "index_documents",
]