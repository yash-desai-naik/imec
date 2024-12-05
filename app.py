import re
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime

# Document processing
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize

# Vector store
from chromadb import PersistentClient, Settings
import chromadb

# Embedding model
from sentence_transformers import SentenceTransformer

# dotenv
from dotenv import load_dotenv
load_dotenv()

# LLM
from groq import Groq

def initialize_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

initialize_nltk()



class DocumentProcessor:
    def __init__(self, docs_dir: str, chunk_size: int = 10000, chunk_overlap: int = 6000):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def get_document_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of document for change detection"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text to standardize article references"""
        # Standardize article references
        text = re.sub(r'(?i)article\s+(\d+)', r'Article \1', text)
        text = re.sub(r'(?i)articles\s+(\d+)', r'Article \1', text)
        return text
    
    def extract_docx_content(self, doc) -> List[Dict]:
        """Extract content while preserving structure and articles"""
        sections = []
        current_section = []
        current_heading = "Default"
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
                
            # Check for article headers
            article_match = re.match(r'(?i)^article\s+(\d+)', text)
            if article_match or paragraph.style.name.startswith('Heading'):
                if current_section:
                    sections.append({
                        'heading': current_heading,
                        'content': '\n'.join(current_section)
                    })
                current_section = []
                current_heading = text
            
            # Preprocess text before adding
            processed_text = self.preprocess_text(text)
            current_section.append(processed_text)

        # Add the last section
        if current_section:
            sections.append({
                'heading': current_heading,
                'content': '\n'.join(current_section)
            })

        return sections

    def process_docx(self, file_path: Path) -> List[Dict]:
        """Process a single DOCX file into chunks with metadata"""
        doc = Document(file_path)
        sections = self.extract_docx_content(doc)
        
        all_chunks = []
        for idx, section in enumerate(sections):
            # Split the content into sentences and create chunks
            sentences = sent_tokenize(section['content'])
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
                # Check if we should create a new chunk
                if current_length >= self.chunk_size:
                    chunk_text = f"Section: {section['heading']}\n" + " ".join(current_chunk)
                    all_chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': str(file_path),
                            'section': section['heading'],
                            'chunk_index': f"{idx}-{len(all_chunks)}",
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    
                    # Keep some sentences for overlap
                    overlap_sentences = current_chunk[-self.chunk_overlap//50:]  # Approximate number of sentences
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)
            
            # Add the last chunk if there's anything left
            if current_chunk:
                chunk_text = f"Section: {section['heading']}\n" + " ".join(current_chunk)
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'source': str(file_path),
                        'section': section['heading'],
                        'chunk_index': f"{idx}-{len(all_chunks)}",
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
        return all_chunks


class RAGSystem:
    def __init__(self, docs_dir: str, db_dir: str):
        self.docs_dir = Path(docs_dir)
        self.db_dir = Path(db_dir)
        self.processor = DocumentProcessor(docs_dir)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure clean directory
        if os.path.exists(str(self.db_dir)):
            import shutil
            shutil.rmtree(str(self.db_dir))
        os.makedirs(str(self.db_dir), exist_ok=True)
        
        # Initialize ChromaDB with specific settings
        settings = Settings(
            is_persistent=True,
            persist_directory=str(self.db_dir),
            anonymized_telemetry=False
        )
        
        try:
            self.chroma_client = PersistentClient(
                path=str(self.db_dir),
                settings=settings
            )
            
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Load or initialize document hashes
        self.hashes_file = self.db_dir / "document_hashes.json"
        self.document_hashes = self._load_document_hashes()
        
    def _load_document_hashes(self) -> Dict[str, str]:
        if self.hashes_file.exists():
            with open(self.hashes_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_document_hashes(self):
        with open(self.hashes_file, 'w') as f:
            json.dump(self.document_hashes, f)
            
    def update_knowledge_base(self):
        current_files = list(self.docs_dir.glob("*.docx"))
        
        for file_path in current_files:
            current_hash = self.processor.get_document_hash(file_path)
            if str(file_path) not in self.document_hashes or self.document_hashes[str(file_path)] != current_hash:
                print(f"Processing updated file: {file_path}")
                
                if str(file_path) in self.document_hashes:
                    self.collection.delete(where={"source": str(file_path)})
                
                chunks = self.processor.process_docx(file_path)
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_model.encode(texts).tolist()
                metadatas = [chunk['metadata'] for chunk in chunks]
                ids = [f"{file_path}_{chunk['metadata']['chunk_index']}" for chunk in chunks]
                
                # Add chunks in smaller batches to handle large documents
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    end_idx = min(i + batch_size, len(texts))
                    self.collection.add(
                        embeddings=embeddings[i:end_idx],
                        documents=texts[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                
                self.document_hashes[str(file_path)] = current_hash
        
        current_file_paths = [str(f) for f in current_files]
        for file_path in list(self.document_hashes.keys()):
            if file_path not in current_file_paths:
                print(f"Removing deleted file: {file_path}")
                self.collection.delete(where={"source": file_path})
                del self.document_hashes[file_path]
        
        self._save_document_hashes()

    def query(self, question: str, n_results: int = 5) -> Dict:
        """Query the RAG system with improved article search"""
        # Preprocess the question
        processor = DocumentProcessor("")  # Empty path as we don't need it here
        processed_question = processor.preprocess_text(question)
        
        # Extract article number if present
        article_match = re.search(r'(?i)article\s+(\d+)', processed_question)
        article_number = article_match.group(1) if article_match else None
        
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode(processed_question).tolist()
        
        # Modify the query based on whether we're searching for a specific article
        if article_number:
            # Use only text search for exact article matches
            results = self.collection.query(
                query_embeddings=[question_embedding],
                where={"$contains": f"Article {article_number}"},
                n_results=n_results
            )
        else:
            # Use semantic search without where clause
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results
            )
        
        # Construct prompt with enhanced context
        context_parts = []
        seen_articles = set()
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            # Check for article references
            article_matches = re.finditer(r'(?i)article\s+(\d+)', doc)
            articles = [m.group(0) for m in article_matches]
            
            # Add to context if new article found or if specifically searching for an article
            if articles:
                for article in articles:
                    if article.lower() not in seen_articles:
                        seen_articles.add(article.lower())
                        section_info = f"Section: {meta['section']}" if 'section' in meta else ""
                        context_parts.append(
                            f"Source: {meta['source']}\n{section_info}\nContent: {doc}"
                        )
            elif article_number:  # Include context if specifically searching for an article
                section_info = f"Section: {meta['section']}" if 'section' in meta else ""
                context_parts.append(
                    f"Source: {meta['source']}\n{section_info}\nContent: {doc}"
                )
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant answering questions based on the provided context.
        When explaining articles, always include the complete article text and explanation.
        Always cite your sources using the filename and section from the context.
        If the specific article or information cannot be found in the context, say so clearly.

        Context:
        {context}

        Question: {processed_question}

        Answer:"""

        # Query Groq
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.2,
        )

        return {
            'answer': response.choices[0].message.content,
            'sources': [f"{meta['source']} (Section: {meta.get('section', 'N/A')})" 
                    for meta in results['metadatas'][0]]
        }

    def get_feedback(self, question: str, answer: str, feedback: str):
        feedback_file = self.db_dir / "feedback.jsonl"
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'feedback': feedback
        }
        
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')


if __name__ == "__main__":
    # Create necessary directories if they don't exist
    DOCS_DIR = os.getenv("DOCS_DIR")
    if not DOCS_DIR:
        raise ValueError("DOCS_DIR environment variable not set")
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # Clean start
    if os.path.exists("./rag_db"):
        import shutil
        print("Cleaning up existing database...")
        shutil.rmtree("./rag_db")
    
    os.makedirs("./rag_db", exist_ok=True)
    
    print("Initializing RAG system...")
    try:
        rag = RAGSystem(
            docs_dir=DOCS_DIR,
            db_dir="./rag_db"
        )
        
        print("Updating knowledge base...")
        rag.update_knowledge_base()
        
        print("Ready for queries!")
        
        # Interactive query loop
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            result = rag.query(question)
            print("\nAnswer:", result['answer'])
            print("\nSources:", result['sources'])
            
            feedback = input("\nWas this answer helpful? (yes/no): ")
            rag.get_feedback(question, result['answer'], feedback)
            
    except Exception as e:
        print(f"Error: {e}")
        print("Cleaning up...")
        if os.path.exists("./rag_db"):
            import shutil
            shutil.rmtree("./rag_db")
        raise