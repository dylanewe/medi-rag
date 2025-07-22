from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PDFUploadResponse(BaseModel):
    """Response model for PDF upload endpoint."""
    success: bool
    message: str
    document_id: str
    chunks_processed: int

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    document_id: Optional[str] = None  # If None, search across all documents

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[dict]
    query_time: float

class ChunkData(BaseModel):
    """Model for chunk data structure."""
    chunk_text: str
    section_title: str
    page_number: int
    document_id: str