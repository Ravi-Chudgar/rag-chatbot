from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.embeddings import get_embeddings
from app.utils import ensure_directory, list_supported_files
from app.vector_store import load_vector_store, vector_store_exists


def _load_documents(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    return TextLoader(str(file_path), encoding="utf-8").load()


def _chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(documents)


def index_files(file_paths: list[Path]) -> tuple[int, int, list[str]]:
    all_documents: list[Document] = []
    for path in file_paths:
        docs = _load_documents(path)
        for doc in docs:
            doc.metadata["source"] = str(path)
        all_documents.extend(docs)

    if not all_documents:
        return 0, 0, []

    chunks = _chunk_documents(all_documents)
    embeddings = get_embeddings()
    index_path = settings.vector_store_path
    ensure_directory(index_path)

    if vector_store_exists(index_path):
        vector_store = load_vector_store(index_path)
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(str(index_path))
    return len(file_paths), len(chunks), [str(path) for path in file_paths]


def index_data_directory() -> tuple[int, int, list[str]]:
    docs_dir = settings.documents_dir
    ensure_directory(docs_dir)
    files = list_supported_files(docs_dir)
    return index_files(files)