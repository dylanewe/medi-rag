import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class Embedder:
    """Handles text embedding using bioBERT model."""
    
    def __init__(self):
        self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading bioBERT model on device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("bioBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load bioBERT model: {e}")
            raise
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks using bioBERT.
        
        Args:
            chunks (List[str]): List of text chunks to embed
            
        Returns:
            List[List[float]]: List of 768-dimensional embedding vectors
        """
        if not chunks:
            return []
        
        embeddings = []
        batch_size = 8  # Process in small batches to avoid memory issues
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        try:
            with torch.no_grad():
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_embeddings = self._embed_batch(batch_chunks)
                    embeddings.extend(batch_embeddings)
                    
                    logger.debug(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _embed_batch(self, batch_chunks: List[str]) -> List[List[float]]:
        """Process a batch of chunks and return their embeddings."""
        # Tokenize the batch
        inputs = self.tokenizer(
            batch_chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token) as sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Convert to list of lists
        return [embedding.tolist() for embedding in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Args:
            query (str): Query text to embed
            
        Returns:
            List[float]: 768-dimensional embedding vector
        """
        if not query:
            return [0.0] * settings.embedding_dimension
        
        try:
            embeddings = self.embed_chunks([query])
            return embeddings[0] if embeddings else [0.0] * settings.embedding_dimension
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise