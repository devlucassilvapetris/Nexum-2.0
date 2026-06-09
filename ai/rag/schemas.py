"""
Pydantic schemas for structured outputs with Instructor
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Document(BaseModel):
    """Document schema"""
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    score: Optional[float] = None


class Answer(BaseModel):
    """Structured answer schema"""
    answer: str = Field(description="The answer to the question")
    sources: List[str] = Field(description="List of source document IDs")
    confidence: float = Field(description="Confidence score from 0 to 1")
    reasoning: Optional[str] = Field(description="Reasoning behind the answer")


class QueryRequest(BaseModel):
    """Query request schema"""
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_rag: bool = Field(default=True)


class QueryResponse(BaseModel):
    """Query response schema"""
    answer: Answer
    retrieved_documents: List[Document]
    query_time: float
