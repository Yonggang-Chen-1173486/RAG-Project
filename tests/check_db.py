import sys
from pathlib import Path
sys.path.insert(0, "src")

from rag.retrieval.vector_store import VectorStore

vs = VectorStore()
print(f"Collection name: {vs.collection_name}")
print(f"Document count: {vs.collection.count()}")

# 看前 3 条（如果有）
if vs.collection.count() > 0:
    sample = vs.collection.get(limit=3)
    for i, (id_, meta) in enumerate(zip(sample["ids"], sample["metadatas"])):
        print(f"  [{i}] id={id_}  source_file={meta.get('source_file')}")