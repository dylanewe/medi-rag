import openai
import json
import logging
from typing import List, Dict, Tuple, Any
import re
from app.config import settings

logger = logging.getLogger(__name__)

class SemanticChunker:
    """Handles semantic sectioning and chunking of medical documents."""
    
    def __init__(self):
        openai.api_key = settings.openai_api_key
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    def identify_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Use GPT-4o-mini to identify medical sections in the document.
        
        Args:
            full_text (str): Complete text of the medical document
            
        Returns:
            List[Dict]: List of sections with titles and starting text
        """
        prompt = """
        You are a medical document analyst. Analyze the following medical document text and identify the major sections.
        
        Return a JSON object with the following structure:
        {
            "sections": [
                {
                    "section_title": "Section Name",
                    "starting_text": "First 50-100 characters of the section",
                    "section_type": "medical_category"
                }
            ]
        }
        
        Common medical sections include but are not limited to:
        - Patient Information/Demographics
        - Chief Complaint
        - History of Present Illness
        - Past Medical History
        - Physical Examination
        - Laboratory Results
        - Diagnostic Tests
        - Assessment/Diagnosis
        - Treatment Plan
        - Medications
        - Follow-up Instructions
        - Discharge Summary
        
        Only return valid JSON. If no clear sections are found, return sections based on content themes.
        
        Document text:
        """.strip()
        
        # Truncate text if too long for API
        max_text_length = 8000  # Conservative limit for GPT-4o-mini
        truncated_text = full_text[:max_text_length]
        if len(full_text) > max_text_length:
            truncated_text += "\n... [Document truncated for analysis]"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical document analyst specialized in identifying document sections."},
                    {"role": "user", "content": f"{prompt}\n\n{truncated_text}"}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"GPT-4o-mini response: {response_text}")
            
            # Parse JSON response
            sections_data = json.loads(response_text)
            sections = sections_data.get("sections", [])
            
            logger.info(f"Identified {len(sections)} sections in the document")
            return sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from GPT-4o-mini: {e}")
            # Fallback: create basic sections
            return self._create_fallback_sections(full_text)
        except Exception as e:
            logger.error(f"Error calling GPT-4o-mini for section identification: {e}")
            return self._create_fallback_sections(full_text)
    
    def _create_fallback_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """Create fallback sections when LLM analysis fails."""
        # Simple heuristic-based section identification
        sections = []
        text_length = len(full_text)
        
        # Create sections based on text length
        if text_length > 3000:
            sections.append({
                "section_title": "Document Beginning",
                "starting_text": full_text[:100],
                "section_type": "introduction"
            })
            
            middle_start = text_length // 2 - 50
            sections.append({
                "section_title": "Document Middle",
                "starting_text": full_text[middle_start:middle_start + 100],
                "section_type": "body"
            })
            
            sections.append({
                "section_title": "Document End",
                "starting_text": full_text[-200:-100] if text_length > 200 else full_text[-100:],
                "section_type": "conclusion"
            })
        else:
            sections.append({
                "section_title": "Complete Document",
                "starting_text": full_text[:100],
                "section_type": "full_document"
            })
        
        logger.warning("Used fallback section identification")
        return sections
    
    def chunk_document(self, document_text: str, sections: List[Dict], 
                      page_mappings: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """
        Chunk the document based on identified sections.
        
        Args:
            document_text (str): Full document text
            sections (List[Dict]): Identified sections from LLM
            page_mappings (List[Tuple]): Page number to text mappings
            
        Returns:
            List[Dict]: List of chunk dictionaries
        """
        chunks = []
        
        # Create a mapping of text positions to page numbers
        text_to_page = self._create_text_to_page_mapping(document_text, page_mappings)
        
        for section in sections:
            section_title = section.get("section_title", "Unknown Section")
            starting_text = section.get("starting_text", "")
            
            # Find section boundaries in the document
            section_start, section_end = self._find_section_boundaries(
                document_text, starting_text, sections, section
            )
            
            if section_start == -1:
                logger.warning(f"Could not locate section: {section_title}")
                continue
            
            section_text = document_text[section_start:section_end]
            
            # Chunk this section
            section_chunks = self._create_chunks(
                section_text, section_title, section_start, text_to_page
            )
            
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _create_text_to_page_mapping(self, document_text: str, 
                                   page_mappings: List[Tuple[int, str]]) -> Dict[int, int]:
        """Create mapping from text position to page number."""
        text_to_page = {}
        current_pos = 0
        
        for page_num, page_text in page_mappings:
            text_start = current_pos
            text_end = current_pos + len(page_text)
            
            # Map all positions in this range to this page
            for pos in range(text_start, text_end):
                text_to_page[pos] = page_num
            
            current_pos = text_end + 1  # Account for spaces between pages
        
        return text_to_page
    
    def _find_section_boundaries(self, document_text: str, starting_text: str, 
                               all_sections: List[Dict], current_section: Dict) -> Tuple[int, int]:
        """Find the start and end positions of a section in the document."""
        if not starting_text:
            return -1, -1
        
        # Find section start
        section_start = document_text.find(starting_text)
        
        if section_start == -1:
            # Try fuzzy matching
            words = starting_text.split()[:5]  # First 5 words
            for i, word in enumerate(words):
                if word in document_text:
                    # Find context around this word
                    word_pos = document_text.find(word)
                    context_start = max(0, word_pos - 100)
                    context_end = min(len(document_text), word_pos + 200)
                    context = document_text[context_start:context_end]
                    
                    if len([w for w in words if w in context]) >= len(words) // 2:
                        section_start = context_start
                        break
        
        if section_start == -1:
            return -1, -1
        
        # Find section end (start of next section or end of document)
        current_index = all_sections.index(current_section)
        
        if current_index < len(all_sections) - 1:
            next_section = all_sections[current_index + 1]
            next_starting_text = next_section.get("starting_text", "")
            section_end = document_text.find(next_starting_text, section_start + 100)
            
            if section_end == -1:
                section_end = len(document_text)
        else:
            section_end = len(document_text)
        
        return section_start, section_end
    
    def _create_chunks(self, section_text: str, section_title: str, 
                      section_start: int, text_to_page: Dict[int, int]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from section text."""
        chunks = []
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        chunk_size_chars = settings.chunk_size * 4
        overlap_chars = settings.chunk_overlap * 4
        
        start = 0
        
        while start < len(section_text):
            end = min(start + chunk_size_chars, len(section_text))
            
            # Try to end at sentence boundary
            if end < len(section_text):
                sentence_end = section_text.rfind('.', start, end)
                if sentence_end > start + chunk_size_chars // 2:
                    end = sentence_end + 1
            
            chunk_text = section_text[start:end].strip()
            
            if chunk_text:
                # Determine page number for this chunk
                chunk_absolute_start = section_start + start
                page_number = text_to_page.get(chunk_absolute_start, 1)
                
                chunk_data = {
                    "chunk_text": chunk_text,
                    "section_title": section_title,
                    "page_number": page_number
                }
                
                chunks.append(chunk_data)
            
            # Move start position with overlap
            start = max(start + chunk_size_chars - overlap_chars, start + 1)
            
            if start >= len(section_text):
                break
        
        return chunks