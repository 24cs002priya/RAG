from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=8, ge=1, le=50)


class Citation(BaseModel):
    document: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    chunk_type: Optional[str] = None
    document_type: Optional[str] = None
    industry: Optional[str] = None
    year: Optional[int] = None
    score: Optional[float] = None


class QAResponse(BaseModel):
    query: str
    answer: str
    sources: List[Citation] = Field(default_factory=list)
    applied_filters: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    filename: str
    message: str
    total_chunks: int
    total_pages: int = 0
    total_tables: int = 0
