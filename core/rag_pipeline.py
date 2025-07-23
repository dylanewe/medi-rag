import time
import logging
from typing import List, Dict, Any, Tuple
import openai
from app.database import DatabaseManager
from core.embedder import Embedder
from app.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline for medical document querying."""
    
    def __init__(self, db_manager: DatabaseManager = None, embedder: Embedder = None):
        self.db_manager = db_manager or DatabaseManager()
        self.embedder = embedder or Embedder()
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        
        logger.info("RAG Pipeline initialized successfully")
    
    def query(self, user_query: str, document_id: str = None) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            user_query (str): User's question
            document_id (str, optional): Specific document to search
            
        Returns:
            Dict: Contains answer, sources, and timing information
        """
        start_time = time.time()
        
        try:
            # Step 1: Embed user query
            logger.info(f"Processing query: {user_query[:100]}...")
            query_embedding = self.embedder.embed_query(user_query)
            
            # Step 2: Vector similarity search
            similar_chunks = self.db_manager.similarity_search(
                query_embedding=query_embedding,
                document_id=document_id,
                limit=settings.max_chunks_retrieval
            )
            
            if not similar_chunks:
                return {
                    "answer": "I couldn't find any relevant information for your query in the medical documents.",
                    "sources": [],
                    "query_time": time.time() - start_time
                }
            
            # Step 3: Re-rank results
            re_ranked_chunks = self._rerank_chunks(similar_chunks, user_query)
            
            # Step 4: Create context from top chunks
            context = self._create_context(re_ranked_chunks[:settings.top_k_context])
            
            # Step 5: Generate final answer using LLM
            answer = self._generate_answer(user_query, context)
            
            # Prepare response
            sources = [
                {
                    "chunk_id": chunk["id"],
                    "section": chunk["section_title"],
                    "page": chunk["page_number"],
                    "similarity_score": round(chunk["similarity_score"], 3),
                    "document_id": chunk["document_id"],
                    "text": chunk["chunk_text"][:100]  # Preview of chunk text
                }
                for chunk in re_ranked_chunks[:settings.top_k_context]
            ]
            
            query_time = time.time() - start_time
            
            logger.info(f"Query processed successfully in {query_time:.2f} seconds")
            
            return {
                "answer": answer,
                "sources": sources,
                "query_time": query_time
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise
    
    def _rerank_chunks(self, chunks: List[Dict], user_query: str) -> List[Dict]:
        """
        Re-rank retrieved chunks using Reciprocal Score Enhancement (RSE).
        
        Args:
            chunks (List[Dict]): Initial retrieved chunks
            user_query (str): Original user query
            
        Returns:
            List[Dict]: Re-ranked chunks
        """
        # Simple RSE implementation based on keyword overlap and similarity score
        query_terms = set(user_query.lower().split())
        
        for chunk in chunks:
            # Base score from vector similarity
            base_score = chunk["similarity_score"]
            
            # Bonus for keyword overlap
            chunk_terms = set(chunk["chunk_text"].lower().split())
            overlap_ratio = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
            keyword_bonus = overlap_ratio * 0.1
            
            # Bonus for medical sections (heuristic)
            section_bonus = 0
            medical_sections = ["diagnosis", "assessment", "treatment", "lab results", "examination"]
            if any(term in chunk["section_title"].lower() for term in medical_sections):
                section_bonus = 0.02
            
            # Calculate final re-ranking score
            chunk["rerank_score"] = base_score + keyword_bonus + section_bonus
        
        # Sort by re-ranking score
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.debug(f"Re-ranked {len(chunks)} chunks")
        return chunks
    
    def _create_context(self, top_chunks: List[Dict]) -> str:
        """
        Create formatted context string from top-ranked chunks.
        
        Args:
            top_chunks (List[Dict]): Top-ranked chunks for context
            
        Returns:
            str: Formatted context string
        """
        if not top_chunks:
            return "No relevant context found."
        
        context_parts = ["CONTEXT FROM MEDICAL DOCUMENT:", ""]
        
        for i, chunk in enumerate(top_chunks, 1):
            section = chunk["section_title"]
            page = chunk["page_number"]
            text = chunk["chunk_text"]
            
            context_part = f"---\n[Chunk {i} from Section: {section}, Page: {page}]: {text}\n---"
            context_parts.append(context_part)
        
        context = "\n".join(context_parts)
        
        # Ensure context isn't too long for the LLM
        max_context_length = 6000  # Conservative limit
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... [Context truncated]"
        
        return context
    
    def _generate_answer(self, user_query: str, context: str) -> str:
        """
        Generate final answer using GPT-4o-mini with medical context.
        
        Args:
            user_query (str): Original user query
            context (str): Formatted context from retrieved chunks
            
        Returns:
            str: Generated answer
        """
        system_prompt = """
        You are a medical expert assistant. Your task is to answer questions about medical documents based ONLY on the provided context.
        
        IMPORTANT GUIDELINES:
        1. Only use information explicitly provided in the context
        2. If the context doesn't contain enough information to answer the question, clearly state this
        3. Be precise and clinical in your language
        4. Reference specific sections/pages when possible
        5. Never make up or assume medical information not in the context
        6. If asked about treatments or diagnoses, only mention what's explicitly stated
        """
        
        user_prompt = f"""
        Based on the following medical document context, please answer this question: {user_query}
        
        {context}
        
        Please provide a comprehensive answer based solely on the information provided above.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer: {answer[:200]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with GPT-4o-mini: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."
