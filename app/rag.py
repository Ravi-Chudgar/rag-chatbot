from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.llm_factory import get_chat_llm
from app.models import SourceChunk
from app.vector_store import load_vector_store, vector_store_exists


class RAGService:
    def __init__(self) -> None:
        self.rag_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use only the context below to answer the question.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
        self.chat_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the question clearly and concisely.\n\n"
            "Question:\n{question}"
        )
        self.llm = get_chat_llm()
        self._cached_vector_store: FAISS | None = None
        self._cached_index_mtime: float | None = None

    @staticmethod
    def _get_index_mtime(index_path: Path) -> float:
        return max((index_path / "index.faiss").stat().st_mtime, (index_path / "index.pkl").stat().st_mtime)

    def _get_cached_vector_store(self, index_path: Path) -> FAISS:
        current_mtime = self._get_index_mtime(index_path)
        if self._cached_vector_store is None or self._cached_index_mtime != current_mtime:
            self._cached_vector_store = load_vector_store(index_path)
            self._cached_index_mtime = current_mtime
        return self._cached_vector_store

    def answer(
        self, question: str, use_documents: bool = False
    ) -> tuple[str, list[SourceChunk], list[str], dict]:
        provider = settings.llm_provider.lower()
        if provider == "ollama":
            model_name = settings.ollama_chat_model
        elif provider == "openai":
            model_name = settings.openai_chat_model
        elif provider in ("claude", "anthropic"):
            model_name = settings.anthropic_chat_model
        elif provider == "huggingface":
            model_name = settings.huggingface_chat_model
        else:
            model_name = "unknown"

        debug_trace: list[str] = [
            "1. Received chat request",
            f"2. Using provider: {provider}, model: {model_name}",
        ]
        rag_process = {
            "mode": "direct",
            "used_documents": bool(use_documents),
            "vector_index_found": False,
            "retrieved_docs": 0,
            "context_chars": 0,
            "steps": [],
        }
        if not use_documents:
            debug_trace.append("3. Document mode is OFF, using direct chat mode")
            rag_process["steps"] = [
                "Query received",
                "Document mode OFF",
                "Direct LLM generation",
                "Response returned",
            ]
            chain = self.chat_prompt | self.llm
            response = chain.invoke({"question": question})
            debug_trace.append("4. Generated response from chat model")
            return response.content, [], debug_trace, rag_process

        index_path = settings.vector_store_path
        debug_trace.append("3. Document mode is ON, checking vector index")
        debug_trace.append(f"4. Checking vector index at: {index_path}")
        rag_process["steps"] = [
            "Query received",
            "Vector DB lookup",
            "Retrieve top chunks",
            "Build context",
            "Generate grounded answer",
            "Return response",
        ]

        if not vector_store_exists(index_path):
            debug_trace.append("5. Vector index not found, switching to direct chat mode")
            rag_process["mode"] = "direct_fallback_no_index"
            chain = self.chat_prompt | self.llm
            response = chain.invoke({"question": question})
            debug_trace.append("6. Generated response from chat model")
            return response.content, [], debug_trace, rag_process

        debug_trace.append("5. Vector index found, loading FAISS store (cache-aware)")
        rag_process["vector_index_found"] = True
        vector_store = self._get_cached_vector_store(index_path)
        docs = vector_store.similarity_search(question, k=settings.top_k)
        debug_trace.append(f"6. Retrieved top-{settings.top_k} results, got {len(docs)} matches")
        rag_process["retrieved_docs"] = len(docs)
        if not docs:
            debug_trace.append("7. No matching documents, switching to direct chat mode")
            rag_process["mode"] = "direct_fallback_no_match"
            chain = self.chat_prompt | self.llm
            response = chain.invoke({"question": question})
            debug_trace.append("8. Generated response from chat model")
            return response.content, [], debug_trace, rag_process

        context = "\n\n".join(doc.page_content for doc in docs)
        debug_trace.append(f"7. Built RAG context with {len(context)} characters")
        rag_process["mode"] = "rag"
        rag_process["context_chars"] = len(context)

        chain = self.rag_prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        debug_trace.append("8. Generated RAG answer from model")

        sources = [
            SourceChunk(
                source=str(doc.metadata.get("source", "unknown")),
                page=doc.metadata.get("page"),
                content=doc.page_content,
            )
            for doc in docs
        ]
        debug_trace.append(f"9. Prepared {len(sources)} source chunks for response")
        return response.content, sources, debug_trace, rag_process