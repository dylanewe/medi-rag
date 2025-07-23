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
        # Configuration for window processing
        self.window_size = 6000  # Characters per window
        self.window_overlap = 1000  # Overlap between windows
    
    def identify_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Use GPT-4o-mini to identify medical sections in the document by processing
        overlapping windows of the full text and merging results.
        
        Args:
            full_text (str): Complete text of the medical document
            
        Returns:
            List[Dict]: Deduplicated list of sections with titles and starting text
        """
        if len(full_text) <= self.window_size:
            # Document is small enough to process in one go
            return self._analyze_single_window(full_text, 0)
        
        # Split into overlapping windows
        windows = self._create_overlapping_windows(full_text)
        all_sections = []
        
        logger.info(f"Processing document in {len(windows)} overlapping windows")
        
        # Process each window
        for i, (window_text, window_start) in enumerate(windows):
            try:
                window_sections = self._analyze_single_window(window_text, window_start)
                all_sections.extend(window_sections)
                logger.debug(f"Window {i+1}/{len(windows)}: Found {len(window_sections)} sections")
            except Exception as e:
                logger.error(f"Error processing window {i+1}: {e}")
                continue
        
        # Merge and deduplicate sections
        merged_sections = self._merge_and_deduplicate_sections(all_sections, full_text)
        
        logger.info(f"Final result: {len(merged_sections)} unique sections after merging")
        return merged_sections
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON from GPT response, handling markdown code blocks.
        
        Args:
            response_text (str): Raw response from GPT
            
        Returns:
            str: Clean JSON string
        """
        # Remove markdown code blocks if present
        if '```json' in response_text:
            # Extract content between ```json and ```
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                # No closing ```, take everything after ```json
                json_text = response_text[start:].strip()
        elif '```' in response_text:
            # Generic code block
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            if end != -1:
                json_text = response_text[start:end].strip()
            else:
                json_text = response_text[start:].strip()
        else:
            # No code blocks, use as is
            json_text = response_text.strip()
        
        return json_text

    def _create_overlapping_windows(self, full_text: str) -> List[Tuple[str, int]]:
        """
        Create overlapping windows from the full text.
        
        Args:
            full_text (str): Complete document text
            
        Returns:
            List[Tuple[str, int]]: List of (window_text, window_start_position) tuples
        """
        windows = []
        start = 0
        
        while start < len(full_text):
            end = min(start + self.window_size, len(full_text))
            
            # Try to end at a sentence or paragraph boundary to avoid cutting mid-sentence
            if end < len(full_text):
                # Look for paragraph breaks first
                paragraph_end = full_text.rfind('\n\n', start, end)
                if paragraph_end > start + self.window_size // 2:
                    end = paragraph_end + 2
                else:
                    # Fall back to sentence boundary
                    sentence_end = full_text.rfind('.', start, end)
                    if sentence_end > start + self.window_size // 2:
                        end = sentence_end + 1
            
            window_text = full_text[start:end].strip()
            if window_text:
                windows.append((window_text, start))
            
            # Move to next window with overlap
            if end >= len(full_text):
                break
            
            start = end - self.window_overlap
            # Ensure we make progress
            if start <= windows[-1][1] if windows else 0:
                start = (windows[-1][1] if windows else 0) + self.window_size // 2
        
        return windows
    
    def _analyze_single_window(self, window_text: str, window_start: int) -> List[Dict[str, Any]]:
        """
        Analyze a single window of text to identify sections.
        
        Args:
            window_text (str): Text window to analyze
            window_start (int): Starting position of this window in the full document
            
        Returns:
            List[Dict]: Sections found in this window
        """
        # Update the prompt to be more explicit about JSON format
        prompt = """
        You are a medical document analyst. Analyze the following medical document text and identify the major sections.
        
        Return ONLY a valid JSON object with the following structure (no markdown, no code blocks, just JSON):
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
        - Vital Signs
        - Social History
        - Family History
        - Review of Systems
        - Procedures
        - Imaging Results
        - Pathology Reports
        - Progress Notes
        
        IMPORTANT: 
        - Only identify sections that are clearly present in the text
        - The starting_text should be the exact beginning of the section as it appears
        - If this appears to be a partial document or continuation, note that in section_type
        - Be precise about section boundaries
        
        Return ONLY valid JSON, no markdown formatting or code blocks.
        
        Document text:
        """.strip()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical document analyst specialized in identifying document sections. Be precise and only identify sections that are clearly present."},
                    {"role": "user", "content": f"{prompt}\n\n{window_text}"}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"GPT-4o-mini response for window at position {window_start}: {response_text}")
            
            # Parse JSON response (handle markdown code blocks)
            json_text = self._extract_json_from_response(response_text)
            sections_data = json.loads(json_text)
            sections = sections_data.get("sections", [])
            
            # Add window context to each section
            for section in sections:
                section['window_start'] = window_start
                section['window_text_start'] = window_text.find(section.get('starting_text', ''))
                section['absolute_position'] = window_start + section['window_text_start']
            
            return sections
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from GPT-4o-mini for window at {window_start}: {e}")
            logger.debug(f"Raw response that failed to parse: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Error calling GPT-4o-mini for window at position {window_start}: {e}")
            return []
    
    def _merge_and_deduplicate_sections(self, all_sections: List[Dict[str, Any]], 
                                      full_text: str) -> List[Dict[str, Any]]:
        """
        Merge sections from multiple windows and remove duplicates.
        
        Args:
            all_sections (List[Dict]): All sections from all windows
            full_text (str): Complete document text for validation
            
        Returns:
            List[Dict]: Deduplicated and merged sections
        """
        if not all_sections:
            return self._create_fallback_sections(full_text)
        
        # Group sections by similarity
        section_groups = self._group_similar_sections(all_sections)
        
        # Merge each group into a single section
        merged_sections = []
        for group in section_groups:
            merged_section = self._merge_section_group(group, full_text)
            if merged_section:
                merged_sections.append(merged_section)
        
        # Sort by position in document
        merged_sections.sort(key=lambda x: x.get('absolute_position', 0))
        
        # Validate and clean up sections
        validated_sections = self._validate_sections(merged_sections, full_text)
        
        return validated_sections
    
    def _group_similar_sections(self, sections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group sections that likely refer to the same section of the document.
        
        Args:
            sections (List[Dict]): All sections to group
            
        Returns:
            List[List[Dict]]: Groups of similar sections
        """
        groups = []
        used_indices = set()
        
        for i, section in enumerate(sections):
            if i in used_indices:
                continue
            
            group = [section]
            used_indices.add(i)
            
            # Find similar sections
            for j, other_section in enumerate(sections):
                if j in used_indices or i == j:
                    continue
                
                if self._are_sections_similar(section, other_section):
                    group.append(other_section)
                    used_indices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_sections_similar(self, section1: Dict[str, Any], section2: Dict[str, Any]) -> bool:
        """
        Check if two sections are likely referring to the same document section.
        
        Args:
            section1, section2 (Dict): Sections to compare
            
        Returns:
            bool: True if sections are similar
        """
        title1 = section1.get('section_title', '').lower().strip()
        title2 = section2.get('section_title', '').lower().strip()
        
        # Exact title match
        if title1 == title2:
            return True
        
        # Similar titles (using word overlap)
        words1 = set(re.findall(r'\w+', title1))
        words2 = set(re.findall(r'\w+', title2))
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            min_words = min(len(words1), len(words2))
            if min_words > 0 and overlap / min_words >= 0.6:
                return True
        
        # Check if starting texts are similar or overlapping
        start1 = section1.get('starting_text', '').strip()
        start2 = section2.get('starting_text', '').strip()
        
        if start1 and start2:
            # Check for substring relationship
            if start1 in start2 or start2 in start1:
                return True
            
            # Check for significant word overlap in starting text
            words1 = set(re.findall(r'\w+', start1.lower()))
            words2 = set(re.findall(r'\w+', start2.lower()))
            
            if len(words1) >= 3 and len(words2) >= 3:
                overlap = len(words1.intersection(words2))
                if overlap >= min(len(words1), len(words2)) * 0.5:
                    return True
        
        return False
    
    def _merge_section_group(self, group: List[Dict[str, Any]], full_text: str) -> Dict[str, Any]:
        """
        Merge a group of similar sections into a single section.
        
        Args:
            group (List[Dict]): Group of similar sections
            full_text (str): Complete document text
            
        Returns:
            Dict: Merged section
        """
        if not group:
            return None
        
        if len(group) == 1:
            return group[0]
        
        # Choose the best section title (longest or most specific)
        best_title = max(group, key=lambda x: len(x.get('section_title', '')))['section_title']
        
        # Choose the best starting text (verify it exists in full document)
        best_starting_text = ""
        best_position = float('inf')
        
        for section in group:
            starting_text = section.get('starting_text', '')
            if starting_text and starting_text in full_text:
                position = full_text.find(starting_text)
                if position != -1 and position < best_position:
                    best_starting_text = starting_text
                    best_position = position
        
        # Determine the best section type
        section_types = [s.get('section_type', '') for s in group if s.get('section_type')]
        best_type = max(set(section_types), key=section_types.count) if section_types else ""
        
        merged_section = {
            'section_title': best_title,
            'starting_text': best_starting_text,
            'section_type': best_type,
            'absolute_position': best_position if best_position != float('inf') else 0,
            'merged_from': len(group)  # For debugging
        }
        
        return merged_section
    
    def _validate_sections(self, sections: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """
        Validate sections and remove any that don't actually exist in the document.
        
        Args:
            sections (List[Dict]): Sections to validate
            full_text (str): Complete document text
            
        Returns:
            List[Dict]: Validated sections
        """
        validated = []
        
        for section in sections:
            starting_text = section.get('starting_text', '')
            
            if not starting_text:
                logger.warning(f"Section '{section.get('section_title', 'Unknown')}' has no starting text")
                continue
            
            if starting_text not in full_text:
                # Try fuzzy matching
                words = starting_text.split()[:5]  # First 5 words
                found_match = False
                
                for word in words:
                    if word in full_text:
                        # Check if enough words from starting_text appear near this word
                        word_pos = full_text.find(word)
                        context = full_text[max(0, word_pos-200):word_pos+300]
                        matching_words = sum(1 for w in words if w in context)
                        
                        if matching_words >= len(words) // 2:
                            found_match = True
                            section['absolute_position'] = word_pos
                            break
                
                if not found_match:
                    logger.warning(f"Section '{section.get('section_title', 'Unknown')}' not found in document")
                    continue
            else:
                section['absolute_position'] = full_text.find(starting_text)
            
            validated.append(section)
        
        return validated
    
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
                "section_type": "introduction",
                "absolute_position": 0
            })
            
            middle_start = text_length // 2 - 50
            sections.append({
                "section_title": "Document Middle",
                "starting_text": full_text[middle_start:middle_start + 100],
                "section_type": "body",
                "absolute_position": middle_start
            })
            
            end_start = max(text_length - 200, text_length // 2)
            sections.append({
                "section_title": "Document End",
                "starting_text": full_text[end_start:end_start + 100] if end_start + 100 < text_length else full_text[end_start:],
                "section_type": "conclusion",
                "absolute_position": end_start
            })
        else:
            sections.append({
                "section_title": "Complete Document",
                "starting_text": full_text[:100],
                "section_type": "full_document",
                "absolute_position": 0
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
        
        # Use absolute position if available
        if 'absolute_position' in current_section:
            section_start = current_section['absolute_position']
        else:
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
        # Sort sections by absolute position
        sorted_sections = sorted([s for s in all_sections if s.get('absolute_position', -1) != -1], 
                               key=lambda x: x.get('absolute_position', 0))
        
        current_pos = current_section.get('absolute_position', section_start)
        section_end = len(document_text)
        
        # Find the next section after current position
        for section in sorted_sections:
            sect_pos = section.get('absolute_position', -1)
            if sect_pos > current_pos:
                section_end = sect_pos
                break
        
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