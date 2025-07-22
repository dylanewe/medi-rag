import fitz  # PyMuPDF
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)

class PDFParser:
    """Handles PDF parsing and text extraction using PyMuPDF."""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF file page by page.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Tuple[int, str]]: List of (page_number, page_text) tuples
            
        Raises:
            Exception: If PDF cannot be opened or processed
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        extracted_pages = []
        
        try:
            # Open PDF document
            pdf_document = fitz.open(pdf_path)
            logger.info(f"Opened PDF with {len(pdf_document)} pages")
            
            # Extract text from each page
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Clean up the text
                    page_text = self._clean_text(page_text)
                    
                    if page_text.strip():  # Only add non-empty pages
                        extracted_pages.append((page_num + 1, page_text))
                        logger.debug(f"Extracted text from page {page_num + 1}")
                    else:
                        logger.warning(f"Page {page_num + 1} appears to be empty")
                        
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            logger.info(f"Successfully extracted text from {len(extracted_pages)} pages")
            
            return extracted_pages
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines with single space, but preserve paragraph breaks
        cleaned_text = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()