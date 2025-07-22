import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""
    
    openai_api_key: str
    database_url: str
    
    # Optional configurations with defaults
    chunk_size: int = 512
    chunk_overlap: int = 100
    max_chunks_retrieval: int = 20
    top_k_context: int = 5
    embedding_dimension: int = 768
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()