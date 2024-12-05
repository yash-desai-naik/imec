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
            chunk_size=1000,     # Larger chunks to capture more context
            chunk_overlap=200,    # Significant overlap
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
                'processed_timestamp': datetime.utcnow().isoformat(),
                'file_path': file_path
            }
            
            # Better section handling
            sections = []
            current_section = {
                'heading': '',
                'content': [],
                'level': 0,
                'parent_heading': ''
            }
            
            # Stack to keep track of parent headings
            heading_stack = []
            
            for paragraph in doc.paragraphs:
                if paragraph.style.name.startswith('Heading'):
                    # Process previous section
                    if current_section['content']:
                        sections.append(current_section.copy())
                    
                    # Get heading level
                    level = int(paragraph.style.name[-1]) if paragraph.style.name[-1].isdigit() else 1
                    
                    # Update heading stack
                    while heading_stack and heading_stack[-1]['level'] >= level:
                        heading_stack.pop()
                    
                    parent_heading = heading_stack[-1]['heading'] if heading_stack else ''
                    
                    # Create new section
                    current_section = {
                        'heading': paragraph.text,
                        'content': [],
                        'level': level,
                        'parent_heading': parent_heading
                    }
                    
                    heading_stack.append({
                        'heading': paragraph.text,
                        'level': level
                    })
                    
                elif paragraph.text.strip():
                    current_section['content'].append(paragraph.text)
            
            # Add final section
            if current_section['content']:
                sections.append(current_section)
            
            # Create chunks with better context
            chunks = []
            seen_content = set()  # For deduplication
            
            for section in sections:
                # Create hierarchical section path
                section_path = f"{section['parent_heading']} > {section['heading']}" if section['parent_heading'] else section['heading']
                
                # Combine content with context
                section_text = f"{section_path}\n\n" if section_path else ""
                section_text += " ".join(section['content'])
                
                # Split into chunks
                section_chunks = self.text_splitter.split_text(section_text)
                
                for i, chunk in enumerate(section_chunks):
                    # Normalize chunk for deduplication
                    normalized_chunk = ' '.join(chunk.split())
                    
                    if normalized_chunk not in seen_content:
                        seen_content.add(normalized_chunk)
                        
                        chunk_metadata = {
                            **metadata,
                            'section': section['heading'],
                            'parent_section': section['parent_heading'],
                            'section_level': section['level'],
                            'chunk_number': i,
                            'total_chunks': len(section_chunks),
                            'chunk_size': self._token_count(chunk)
                        }
                        
                        chunks.append({
                            'text': chunk,
                            'metadata': chunk_metadata
                        })
            
            logger.info(f"Created {len(chunks)} unique chunks with improved context")
            return {
                'doc_id': doc_id,
                'chunks': chunks,
                'metadata': metadata
            }
            
        except Exception as e:
         logger.error(f"Error processing document: {str(e)}", exc_info=True)
         raise