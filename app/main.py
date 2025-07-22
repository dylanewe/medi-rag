import os
import uuid
import tempfile
import asyncio
import logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models import PDFUploadResponse, QueryRequest, QueryResponse
from app.database import DatabaseManager
from core.pdf_parser import PDFParser
from core.semantic_chunker import SemanticChunker
from core.embedder import Embedder
from core.rag_pipeline import RAGPipeline
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG System",
    description="Retrieval-Augmented Generation system for medical PDF documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
db_manager = DatabaseManager()
pdf_parser = PDFParser()
semantic_chunker = SemanticChunker()
embedder = Embedder()
rag_pipeline = RAGPipeline(db_manager, embedder)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting Medical RAG System...")
    try:
        # Ensure database tables exist
        db_manager.create_tables()
        logger.info("Medical RAG System started successfully")
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Medical RAG System...")
    db_manager.close()

async def process_pdf_background(pdf_path: str, document_id: str) -> dict:
    """
    Background task to process uploaded PDF through the complete pipeline.
    
    Args:
        pdf_path (str): Path to the uploaded PDF file
        document_id (str): Unique identifier for the document
        
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Starting background processing for document {document_id}")
        
        # Step 1: Extract text from PDF
        logger.info("Extracting text from PDF...")
        page_texts = pdf_parser.extract_text_from_pdf(pdf_path)
        full_text = " ".join([page_text for _, page_text in page_texts])
        
        # Step 2: Identify sections using LLM
        logger.info("Identifying document sections...")
        sections = semantic_chunker.identify_sections(full_text)
        
        # Step 3: Create chunks
        logger.info("Creating document chunks...")
        chunks = semantic_chunker.chunk_document(full_text, sections, page_texts)
        
        if not chunks:
            raise ValueError("No chunks were created from the document")
        
        # Step 4: Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = embedder.embed_chunks(chunk_texts)
        
        # Step 5: Prepare data for database insertion
        chunks_data = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data = {
                "document_id": document_id,
                "chunk_text": chunk["chunk_text"],
                "section_title": chunk["section_title"],
                "page_number": chunk["page_number"],
                "embedding": embedding
            }
            chunks_data.append(chunk_data)
        
        # Step 6: Insert into database
        logger.info("Inserting chunks into database...")
        inserted_count = db_manager.insert_chunks(chunks_data)
        
        logger.info(f"Successfully processed document {document_id}: {inserted_count} chunks inserted")
        
        return {
            "success": True,
            "chunks_processed": inserted_count,
            "sections_identified": len(sections)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {document_id}: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary file {pdf_path}: {e}")

@app.post("/upload_pdf/", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a medical PDF document.
    
    Args:
        file: PDF file to be processed
        
    Returns:
        PDFUploadResponse: Upload confirmation with document ID
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Uploaded PDF saved temporarily: {tmp_file_path}")
        
        # Add background task for processing
        background_tasks.add_task(process_pdf_background, tmp_file_path, document_id)
        
        return PDFUploadResponse(
            success=True,
            message=f"PDF uploaded successfully. Processing started for document {document_id}",
            document_id=document_id,
            chunks_processed=0  # Will be updated after background processing
        )
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the medical documents using RAG pipeline.
    
    Args:
        request: Query request containing question and optional document ID
        
    Returns:
        QueryResponse: Generated answer with sources and timing
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Execute RAG pipeline
        result = rag_pipeline.query(
            user_query=request.query,
            document_id=request.document_id
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query_time=result["query_time"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Medical RAG System is running"}

@app.get("/documents")
async def list_documents():
    """Get list of processed documents."""
    try:
        with db_manager.connection.cursor() as cursor:
            cursor.execute("""
                SELECT document_id, 
                       COUNT(*) as chunk_count,
                       MIN(created_at) as uploaded_at
                FROM medical_chunks 
                GROUP BY document_id 
                ORDER BY uploaded_at DESC
            """)
            
            documents = cursor.fetchall()
            return {"documents": [dict(doc) for doc in documents]}
            
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )