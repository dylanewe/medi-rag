# Medical RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system specifically designed for parsing and querying large medical PDF documents using semantic sectioning, bioBERT embeddings, and PostgreSQL with pgvector.

## Features

- **PDF Processing**: Efficient text extraction from medical PDFs using PyMuPDF
- **Semantic Sectioning**: Automated identification of medical document sections using GPT-4o-mini
- **Medical Embeddings**: Uses bioBERT (dmis-lab/biobert-base-cased-v1.1) for domain-specific text embeddings
- **Vector Search**: PostgreSQL with pgvector for efficient similarity search
- **Re-ranking**: Reciprocal Score Enhancement (RSE) for improved retrieval accuracy
- **RESTful API**: FastAPI-based web service for document upload and querying

## Project Structure

```
medical-rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI application and endpoints
│   ├── database.py         # PostgreSQL connection and operations
│   ├── models.py           # Pydantic models for API
│   └── config.py           # Configuration management
├── core/
│   ├── __init__.py
│   ├── pdf_parser.py       # PDF text extraction
│   ├── semantic_chunker.py # Document sectioning and chunking
│   ├── embedder.py         # bioBERT embedding generation
│   └── rag_pipeline.py     # Complete RAG query pipeline
├── scripts/
│   ├── __init__.py
│   └── setup_database.py   # Database initialization script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from template)
└── README.md              # This file
```

## Prerequisites

- Python 3.8+
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- CUDA-compatible GPU (recommended for bioBERT)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd medical-rag-system
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL with pgvector:**
   ```sql
   -- Install pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   
   -- Create database
   CREATE DATABASE medical_rag_db;
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/medical_rag_db
   ```

5. **Initialize the database:**
   ```bash
   python scripts/setup_database.py
   ```

## Usage

### Starting the API Server

```bash
# Development mode with auto-reload
python app/main.py

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

### API Endpoints

#### 1. Upload PDF Document
```bash
curl -X POST "http://localhost:8000/upload_pdf/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@medical_document.pdf"
```

Response:
```json
{
  "success": true,
  "message": "PDF uploaded successfully. Processing started for document abc-123",
  "document_id": "abc-123",
  "chunks_processed": 0
}
```

#### 2. Query Documents
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the patient's diagnosis?",
    "document_id": "abc-123"
  }'
```

Response:
```json
{
  "answer": "Based on the medical document, the patient was diagnosed with...",
  "sources": [
    {
      "chunk_id": 1,
      "section": "Diagnosis",
      "page": 3,
      "similarity_score": 0.892,
      "document_id": "abc-123"
    }
  ],
  "query_time": 1.234
}
```

#### 3. List Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

#### 4. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

## System Architecture

### Processing Pipeline

1. **PDF Upload**: Document uploaded via REST API
2. **Text Extraction**: PyMuPDF extracts text page by page
3. **Semantic Sectioning**: GPT-4o-mini identifies medical sections
4. **Chunking**: Text split into overlapping chunks with section context
5. **Embedding**: bioBERT generates 768-dim vectors for each chunk
6. **Storage**: Chunks and embeddings stored in PostgreSQL with pgvector

### Query Pipeline

1. **Query Embedding**: User query embedded with bioBERT
2. **Vector Search**: PostgreSQL pgvector finds similar chunks
3. **Re-ranking**: RSE algorithm improves result relevance
4. **Context Creation**: Top chunks formatted for LLM
5. **Answer Generation**: GPT-4o-mini generates final answer

## Configuration

Key configuration options in `app/config.py`:

- `chunk_size`: Token size for text chunks (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 100)
- `max_chunks_retrieval`: Initial retrieval count (default: 20)
- `top_k_context`: Chunks used for LLM context (default: 5)
- `embedding_dimension`: bioBERT vector dimension (768)

## Performance Optimization

### Database Indexing
The system creates an IVFFlat index on embeddings for fast similarity search:
```sql
CREATE INDEX medical_chunks_embedding_idx 
ON medical_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### GPU Acceleration
bioBERT automatically uses CUDA if available. Monitor GPU usage:
```bash
nvidia-smi
```

### Batch Processing
- Embeddings generated in batches of 8 chunks
- Database insertions use bulk operations
- PDF processing runs as background tasks

## Troubleshooting

### Common Issues

1. **pgvector not installed:**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-14-pgvector
   
   # Or compile from source
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

2. **CUDA out of memory:**
   - Reduce batch size in `embedder.py`
   - Use CPU instead: remove CUDA detection

3. **OpenAI API errors:**
   - Check API key validity
   - Monitor usage limits
   - Verify model availability

4. **Database connection errors:**
   - Verify DATABASE_URL format
   - Check PostgreSQL service status
   - Ensure database exists

### Logging

The system provides comprehensive logging. Set log level in environment:
```bash
export LOG_LEVEL=DEBUG
python app/main.py
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs for error details
3. Open an issue on the repository
4. Consult the API documentation at `/docs`

## Acknowledgments

- **bioBERT**: Developed by DMIS Lab
- **pgvector**: PostgreSQL vector similarity search
- **FastAPI**: Modern web framework for APIs
- **PyMuPDF**: Efficient PDF processing
"""
