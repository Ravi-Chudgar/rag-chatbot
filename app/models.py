from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query")
    use_documents: bool = Field(
        default=False,
        description="When true, use uploaded/indexed documents for RAG",
    )


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    debug_trace: list[str] = Field(default_factory=list)
    rag_process: dict[str, Any] = Field(default_factory=dict)
    execution_flow: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    indexed_files: int
    indexed_chunks: int
    files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)