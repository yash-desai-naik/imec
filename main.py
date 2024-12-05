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
import time

load_dotenv()

# Setup logging
logger = setup_logger('main')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Process documents on startup."""
    try:
        # Cleanup existing index
        if os.getenv('PINECONE_INDEX_NAME') in vector_store.pc.list_indexes().names():
            logger.info("Cleaning up existing index")
            vector_store.pc.delete_index(os.getenv('PINECONE_INDEX_NAME'))
            
        # Wait for deletion to complete
        time.sleep(5)
        
        logger.info("Processing documents on startup")
        results = doc_processor.process_all_documents()
        
        # Index documents
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
