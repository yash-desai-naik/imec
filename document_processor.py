import re
import os
from typing import Dict, List, Optional, Tuple
from docx import Document
import tiktoken
from datetime import datetime
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import setup_logger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = setup_logger('document_processor')

class DocumentProcessor:
    def __init__(self):
        self.docs_dir = os.getenv('DOCS_DIR')
        if not self.docs_dir:
            raise ValueError("DOCS_DIR environment variable not set")
        
        logger.info(f"Initializing DocumentProcessor with docs directory: {self.docs_dir}")
        os.makedirs(self.docs_dir, exist_ok=True)
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Smaller chunks for more precise matching
            chunk_overlap=100,   # Good overlap for context
            length_function=self._token_count,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _generate_doc_id(self, file_name: str) -> str:
        """Generate a unique document ID based on filename and timestamp."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{os.path.splitext(file_name)[0]}_{timestamp}"
    
    def process_all_documents(self) -> List[Dict]:
        """Process all .docx files in the docs directory."""
        logger.info("Processing all documents in docs directory")
        results = []
        
        try:
            for filename in os.listdir(self.docs_dir):
                if filename.endswith('.docx'):
                    file_path = os.path.join(self.docs_dir, filename)
                    try:
                        result = self.process_document(file_path)
                        results.append(result)
                        logger.info(f"Successfully processed {filename}")
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        continue
            
            logger.info(f"Processed {len(results)} documents successfully")
            return results
        
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        # In document_processor.py
    def process_document(self, file_path: str) -> Dict:
        try:
            doc = Document(file_path)
            doc_id = self._generate_doc_id(os.path.basename(file_path))
            
            metadata = {
                'doc_id': doc_id,
                'filename': os.path.basename(file_path),
                'title': 'IBF-AMOSUP/IMEC AGREEMENT 2024-2025',
                'processed_timestamp': datetime.utcnow().isoformat()
            }
            
            # Collect all text first
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            # Join the full text and split into potential articles
            text = '\n'.join(full_text)
            # Split on Article or ART. pattern
            potential_articles = re.split(r'(?i)(?=(?:Article|ART\.)\s+\d+[:\s])', text)
            
            chunks = []
            for article_text in potential_articles:
                if not article_text.strip():
                    continue
                    
                # Extract article number and title
                article_match = re.match(r'(?i)(?:Article|ART\.)\s+(\d+)[:\s]*(.*?)(?:\n|$)', article_text)
                if article_match:
                    article_num = article_match.group(1)
                    article_title = article_match.group(2).strip()
                    article_content = article_text[article_match.end():].strip()
                    
                    # Create chunk with complete article context
                    chunk_text = f"Article {article_num}: {article_title}\n\n{article_content}"
                    
                    # Split large articles into smaller chunks while preserving context
                    if len(chunk_text) > 1000:
                        # Split into smaller chunks
                        sub_chunks = self.text_splitter.split_text(article_content)
                        for i, sub_chunk in enumerate(sub_chunks):
                            chunk_metadata = {
                                **metadata,
                                'article_number': str(article_num),
                                'article_title': article_title or "Untitled",
                                'is_partial': True,
                                'part_number': i + 1,
                                'total_parts': len(sub_chunks),
                                'chunk_type': 'article_section'
                            }
                            
                            # Include article context in each chunk
                            formatted_chunk = (
                                f"Article {article_num}: {article_title}\n"
                                f"Part {i + 1} of {len(sub_chunks)}\n\n"
                                f"{sub_chunk}"
                            )
                            
                            chunks.append({
                                'text': formatted_chunk,
                                'metadata': chunk_metadata
                            })
                    else:
                        chunk_metadata = {
                            **metadata,
                            'article_number': str(article_num),
                            'article_title': article_title or "Untitled",
                            'is_partial': False,
                            'chunk_type': 'complete_article'
                        }
                        
                        chunks.append({
                            'text': chunk_text,
                            'metadata': chunk_metadata
                        })
            
            logger.info(f"Created {len(chunks)} chunks from document")
            return {
                'doc_id': doc_id,
                'chunks': chunks,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise