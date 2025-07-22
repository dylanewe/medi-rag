import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                settings.database_url,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database successfully")
            self._enable_pgvector()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _enable_pgvector(self):
        """Enable pgvector extension if not already enabled."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("pgvector extension enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable pgvector extension: {e}")
            raise
    
    def create_tables(self):
        """Create the medical_chunks table with proper schema."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS medical_chunks (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            chunk_text TEXT NOT NULL,
            section_title VARCHAR(255),
            page_number INTEGER,
            embedding vector(768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        create_index_query = """
        CREATE INDEX IF NOT EXISTS medical_chunks_embedding_idx 
        ON medical_chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_query)
                cursor.execute(create_index_query)
                logger.info("Tables and indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def insert_chunks(self, chunks_data: List[Dict[str, Any]]) -> int:
        """Insert multiple chunks with their embeddings into the database."""
        insert_query = """
        INSERT INTO medical_chunks (document_id, chunk_text, section_title, page_number, embedding)
        VALUES %s
        """
        
        try:
            with self.connection.cursor() as cursor:
                values = [
                    (
                        chunk['document_id'],
                        chunk['chunk_text'],
                        chunk['section_title'],
                        chunk['page_number'],
                        chunk['embedding']
                    )
                    for chunk in chunks_data
                ]
                
                execute_values(cursor, insert_query, values, template=None, page_size=100)
                logger.info(f"Inserted {len(chunks_data)} chunks into database")
                return len(chunks_data)
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], 
                         document_id: str = None, limit: int = 20) -> List[Dict]:
        """Perform vector similarity search using pgvector."""
        base_query = """
        SELECT id, document_id, chunk_text, section_title, page_number,
               1 - (embedding <=> %s::vector) as similarity_score
        FROM medical_chunks
        """
        
        if document_id:
            query = base_query + " WHERE document_id = %s ORDER BY embedding <=> %s::vector LIMIT %s"
            params = (query_embedding, document_id, query_embedding, limit)
        else:
            query = base_query + " ORDER BY embedding <=> %s::vector LIMIT %s"
            params = (query_embedding, query_embedding, limit)
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")