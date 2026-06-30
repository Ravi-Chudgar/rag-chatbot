from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from app.embeddings import get_embeddings


def load_vector_store(index_path: Path) -> FAISS:
    return FAISS.load_local(
        folder_path=str(index_path),
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def vector_store_exists(index_path: Path) -> bool:
    return (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists()


def get_vector_store_overview(index_path: Path) -> dict[str, Any]:
    if not vector_store_exists(index_path):
        return {
            "exists": False,
            "path": str(index_path),
            "total_vectors": 0,
            "total_chunks": 0,
            "source_count": 0,
            "sources": [],
            "files": [],
        }

    store = load_vector_store(index_path)
    documents = list(getattr(store.docstore, "_dict", {}).values())
    sources = sorted(
        {
            str(doc.metadata.get("source", "unknown"))
            for doc in documents
            if hasattr(doc, "metadata")
        }
    )
    files = []
    for name in ("index.faiss", "index.pkl"):
        file_path = index_path / name
        if file_path.exists():
            files.append({"name": name, "size_bytes": file_path.stat().st_size})

    return {
        "exists": True,
        "path": str(index_path),
        "total_vectors": int(store.index.ntotal),
        "total_chunks": len(documents),
        "source_count": len(sources),
        "sources": sources,
        "files": files,
    }