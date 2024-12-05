# FILE_NAME: /Users/yashdesai/Desktop/work/imec/vector_store.py
```
import os
from typing import List, Dict, Optional
from embeddings import CustomEmbeddings
from pinecone import Pinecone, ServerlessSpec
from utils.logger import setup_logger
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger('vector_store')

class VectorStoreManager:
    def __init__(self):
        """Initialize vector store with Pinecone."""
        try:
            logger.info("Initializing VectorStoreManager")
            
            # Initialize embeddings
            self.embeddings = CustomEmbeddings()
            
            # Initialize Pinecone
            self.pc = Pinecone(
                api_key=os.getenv('PINECONE_API_KEY')
            )
            
            index_name = os.getenv('PINECONE_INDEX_NAME', 'imec-qa')
            
            # Delete index if it exists with wrong dimensions
            if index_name in self.pc.list_indexes().names():
                try:
                    existing_index = self.pc.Index(index_name)
                    existing_desc = self.pc.describe_index(index_name)
                    if existing_desc.dimension != self.embeddings.embedding_size:
                        logger.info(f"Deleting index {index_name} as dimensions don't match (existing: {existing_desc.dimension}, required: {self.embeddings.embedding_size})")
                        self.pc.delete_index(index_name)
                except Exception as e:
                    logger.error(f"Error checking existing index: {str(e)}")
                    raise
            
            # Create index if it doesn't exist
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=self.embeddings.embedding_size,  
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=os.getenv('PINECONE_ENV', 'us-east-1')
                    )
                )
            
            self.index = self.pc.Index(index_name)
            logger.info("VectorStoreManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VectorStoreManager: {str(e)}", exc_info=True)
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info("Embeddings generated successfully")
            return embeddings
        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                logger.error("Embeddings API rate limit exceeded. Please wait a few minutes before trying again.")
                raise Exception("Embeddings API rate limit exceeded. Please wait a few minutes before trying again.") from e
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store."""
        try:
            logger.info(f"Adding {len(texts)} texts to vector store")
            
            # Generate embeddings
            embeddings = self._get_embeddings(texts)
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return []
            
            # Prepare vectors for upsertion
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vector_id = f"vec_{len(vectors)}_{hash(text)}"
                metadata = metadatas[i] if metadatas else {"text": text}
                metadata["text"] = text  # Always include source text in metadata
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert to Pinecone in batches
            batch_size = 100
            total_vectors = len(vectors)
            logger.info(f"Upserting {total_vectors} vectors to Pinecone in batches of {batch_size}")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.error(f"Error upserting batch: {str(e)}")
                    raise
            
            logger.info(f"Successfully added {len(texts)} texts to vector store")
            return [v['id'] for v in vectors]
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}", exc_info=True)
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        try:
            logger.info(f"Performing similarity search for query: {query}")
            
            query_embeddings = self._get_embeddings([query])
            if not query_embeddings:
                logger.error("Failed to generate query embeddings")
                return []
            
            results = self.index.query(
                vector=query_embeddings[0],
                top_k=k * 4,
                include_metadata=True,
                include_values=False
            )
            
            docs = []
            seen_texts = set()  # For deduplication
            
            if results.matches:
                logger.info(f"Found {len(results.matches)} initial results")
                all_scores = [match.score for match in results.matches]
                logger.info(f"All similarity scores: {all_scores}")
                
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                threshold = max(0.3, avg_score * 0.8)
                
                logger.info(f"Using similarity threshold: {threshold}")
                
                for match in results.matches:
                    if match.score >= threshold:
                        if match.metadata and 'text' in match.metadata:
                            # Create a normalized version of text for deduplication
                            normalized_text = ' '.join(match.metadata['text'].split())
                            
                            if normalized_text not in seen_texts:
                                seen_texts.add(normalized_text)
                                
                                doc = {
                                    'text': match.metadata['text'],
                                    'metadata': {
                                        'source': match.metadata.get('source', 'Unknown'),
                                        'score': match.score,
                                        'page_number': match.metadata.get('page_number', 'Unknown'),
                                        'section': match.metadata.get('section', 'Unknown'),
                                        'section_level': match.metadata.get('section_level', 0)
                                    }
                                }
                                
                                logger.info(f"Including unique match with score: {match.score}")
                                logger.info(f"Section: {doc['metadata']['section']}")
                                logger.info(f"Text preview: {doc['text'][:200]}...")
                                
                                docs.append(doc)
            
            # Sort by score and take top k unique results
            docs = sorted(docs, key=lambda x: x['metadata']['score'], reverse=True)[:k]
            
            logger.info(f"Returning {len(docs)} unique documents after deduplication")
            return docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            raise```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/rag_engine.py
```
# rag_engine.py
import re
import os
from typing import Dict, List
from groq import Groq
from utils.logger import setup_logger
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger('rag_engine')

class RAGQueryEngine:
    def __init__(self, vector_store):
        """Initialize RAG Query Engine."""
        try:
            logger.info("Initializing RAG Query Engine")
            self.vector_store = vector_store
            self.model = "mixtral-8x7b-32768"
            self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            logger.info("RAG Query Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Query Engine: {str(e)}", exc_info=True)
            raise

    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query for better matching."""
        # Normalize article references
        query = re.sub(r'(?i)art\.?\s*(\d+)', r'article \1', query)
        query = re.sub(r'(?i)article\s+(\d+)', r'Article \1', query)
        
        # Expand common abbreviations
        query = query.replace('exp', 'explain')
        query = query.replace('desc', 'describe')
        
        logger.info(f"Preprocessed query: {query}")
        return query

    def process_query(self, query: str) -> str:
        """Process a query using RAG."""
        try:
            logger.info(f"Processing query: {query}")
            processed_query = self._preprocess_query(query)
            
            # Get relevant documents
            docs = self.vector_store.similarity_search(processed_query, k=4)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Create a more structured context with better formatting
            context_parts = []
            for doc in docs:
                # Extract meaningful metadata
                section = doc['metadata'].get('section', '').strip()
                page = doc['metadata'].get('page_number', '')
                score = doc['metadata'].get('score', 0)
                
                # Only include high-confidence matches
                if score >= 0.5:  # Increased confidence threshold
                    context = f"""
                    {'Section: ' + section if section else ''}
                    {'Page: ' + str(page) if page else ''}
                    Content: {doc['text'].strip()}
                    """
                    context_parts.append(context.strip())
            
            if not context_parts:
                return "I couldn't find any sufficiently relevant information to answer your question accurately. Could you please rephrase or be more specific?"
            
            # Create a more focused prompt
            line_separator = "-" * 80
            context_text = ("\n" + line_separator + "\n").join(context_parts)

            prompt = (
                "You are an expert assistant analyzing a legal or business document. "
                "Answer the question using only the information from the provided context. "
                "Be specific and detail-oriented.\n\n"
                f"Context (from most to least relevant):\n{line_separator}\n{context_text}\n{line_separator}\n\n"
                f"Question: {processed_query}\n\n"
                "Instructions:\n"
                "1. Focus only on the information present in the context\n"
                "2. Quote relevant parts of the text when appropriate\n"
                "3. If the context doesn't contain enough information, say so clearly\n"
                "4. If referenced articles/sections are mentioned in the context, include their details\n"
                "5. Maintain a professional, precise tone\n"
                "6. Format the response clearly with proper paragraph breaks\n\n"
                "Detailed answer:"
            )
            
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a knowledgeable legal/business document analyst. 
                            Your responses should be:
                            - Precise and specific to the context
                            - Well-structured and clear
                            - Professional in tone
                            - Based solely on the provided information
                            Always quote relevant portions of the text to support your explanations."""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.2,  # Lower temperature for more focused responses
                    max_tokens=1000
                )
                
                answer = chat_completion.choices[0].message.content
                logger.info("Generated response from LLM")
                return answer
                
            except Exception as e:
                logger.error(f"Error from Groq API: {str(e)}")
                return "I apologize, but I encountered an error while processing your question. Please try again."
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/main.py
```
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGQueryEngine
from utils.logger import setup_logger
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, StreamingResponse  # Add StreamingResponse here
import asyncio
load_dotenv()

# Setup logging
logger = setup_logger('main')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Process documents on startup."""
    try:
        logger.info("Processing documents on startup")
        results = doc_processor.process_all_documents()
        
        # Index all documents in vector store
        for doc_result in results:
            chunks = doc_result['chunks']
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        logger.info("Successfully processed and indexed all documents")
    except Exception as e:
        logger.error(f"Error during startup document processing: {str(e)}", exc_info=True)
        raise
    yield
# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    question: str

# Initialize components
try:
    logger.info("Initializing system components")
    doc_processor = DocumentProcessor()
    vector_store = VectorStoreManager()
    rag_engine = RAGQueryEngine(vector_store)
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}", exc_info=True)
    raise





@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page."""
    logger.info("Serving index page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stream-query")
async def stream_query(question: str):
    """Stream the query response."""
    async def generate_response():
        try:
            # Process the query
            logger.info(f"Processing streaming query: {question}")
            response = rag_engine.process_query(question)
            
            # Split response into smaller chunks for streaming
            # This creates a more natural reading experience
            words = response.split()
            chunks = []
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                if len(current_chunk) >= 5:  # Stream 5 words at a time
                    yield f"data: {' '.join(current_chunk)}\n\n"
                    current_chunk = []
                    await asyncio.sleep(0.1)  # Small delay for natural reading
            
            # Send any remaining words
            if current_chunk:
                yield f"data: {' '.join(current_chunk)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
            yield f"data: Error processing query: {str(e)}\n\n"
        
        # Send end of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )

@app.post("/query")
async def query(query: Query):
    """Process a query and return the response."""
    try:
        logger.info(f"Processing query: {query.question}")
        response = rag_engine.process_query(query.question)
        logger.info("Query processed successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/embeddings.py
```
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os
from utils.logger import setup_logger

logger = setup_logger('embeddings')

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embeddings with MiniLM model."""
        self.model = SentenceTransformer(model_name)
        self.embedding_size = 384  # all-MiniLM-L6-v2 embedding size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            return self.model.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}", exc_info=True)
            raise
```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/document_processor.py
```
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
         raise```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/utils/logger.py
```
import logging
import sys
from datetime import datetime

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # File handler
    file_handler = logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/.env
```
# API Keys
GROQ_API_KEY=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
PINECONE_API_KEY=pcsk_5mTPvV_UevhnrgE93EGkEK2yzX6GuVwNsPmpmBykvujauC9ZFN31Eoi5c4mBG38HYLuKKK

# Pinecone Configuration
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=imec-qa

# AWS Credentials
AWS_ACCESS_KEY_ID=AKIAZKUC42ASKWM72X36
AWS_SECRET_ACCESS_KEY=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

# LangChain Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=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
LANGCHAIN_PROJECT=pr-frosty-kitty-3

# Document Directory
DOCS_DIR=/Users/yashdesai/Desktop/docs
```
-----
# FILE_NAME: /Users/yashdesai/Desktop/work/imec/templates/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMEC AI QnA</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold text-center mb-8">IMEC AI QnA</h1>
        
        <!-- Query Section -->
        <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold mb-4">Ask a Question</h2>
            <div class="space-y-4">
                <div>
                    <textarea id="question" rows="3" class="w-full p-2 border rounded-md" placeholder="Enter your question..."></textarea>
                </div>
                <button onclick="submitQuery()" class="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 transition-colors">
                    Submit Query
                </button>
            </div>
            
            <!-- Response Section -->
            <div id="response" class="mt-6 hidden">
                <h3 class="font-semibold mb-2">Answer:</h3>
                <div id="answer" class="p-4 bg-gray-50 rounded-md"></div>
                
                <div id="sources" class="mt-4">
                    <h3 class="font-semibold mb-2">Sources:</h3>
                    <ul id="sourcesList" class="list-disc pl-5 space-y-2"></ul>
                </div>
            </div>
            
            <!-- Loading Indicator -->
            <div id="loading" class="hidden mt-4 text-center">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p class="mt-2 text-gray-600">Processing your query...</p>
            </div>
        </div>
    </div>

    <!-- In index.html, update the script section -->
<script>
    let currentStream = null;

    async function submitQuery() {
        const question = document.getElementById('question').value.trim();
        if (!question) return;

        // Show loading, hide response
        const loading = document.getElementById('loading');
        const response = document.getElementById('response');
        const answer = document.getElementById('answer');
        
        loading.classList.remove('hidden');
        response.classList.add('hidden');
        
        // Close any existing stream
        if (currentStream) {
            currentStream.close();
        }

        try {
            // Try streaming first
            currentStream = new EventSource(`/stream-query?question=${encodeURIComponent(question)}`);
            let fullResponse = '';

            currentStream.onmessage = (event) => {
                if (event.data === '[DONE]') {
                    currentStream.close();
                    loading.classList.add('hidden');
                    return;
                }

                fullResponse += event.data + ' ';
                answer.textContent = fullResponse;
                response.classList.remove('hidden');
            };

            currentStream.onerror = async (error) => {
                // If streaming fails, fall back to regular query
                currentStream.close();
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        answer.textContent = data.response;
                        document.getElementById('response').classList.remove('hidden');
                    } else {
                        throw new Error(data.detail || 'Unknown error');
                    }
                } catch (error) {
                    answer.textContent = `Error: ${error.message}`;
                    response.classList.remove('hidden');
                }
            };

        } catch (error) {
            answer.textContent = `Error: ${error.message}`;
            response.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    }

    // Cleanup on page unload
    window.onbeforeunload = () => {
        if (currentStream) {
            currentStream.close();
        }
    };
</script>
</body>
</html>
```
-----
