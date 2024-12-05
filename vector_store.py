import os
from typing import List, Dict, Optional
from embeddings import CustomEmbeddings
from pinecone import Pinecone, ServerlessSpec
from utils.logger import setup_logger
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger('vector_store')

# In vector_store.py
class VectorStoreManager:
  
    def __init__(self):
        """Initialize vector store with Pinecone."""
        try:
            logger.info("Initializing VectorStoreManager")
            
            # Initialize embeddings
            self.embeddings = CustomEmbeddings()
            
            # Get environment variables
            api_key = os.getenv('PINECONE_API_KEY')
            env = os.getenv('PINECONE_ENV')
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            if not all([api_key, env, index_name]):
                raise ValueError("Missing required Pinecone configuration")
            
            logger.info(f"Connecting to Pinecone environment: {env}")
            
            # Initialize Pinecone with strict environment config
            self.pc = Pinecone(
                api_key=api_key,
                environment=env  # Using PINECONE_ENV from .env
            )
            
            # Delete index if it exists with wrong dimensions
            if index_name in self.pc.list_indexes().names():
                try:
                    existing_index = self.pc.describe_index(index_name)
                    if existing_index.dimension != self.embeddings.embedding_size:
                        logger.info(f"Deleting index {index_name} as dimensions don't match")
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
                        region=env  # Using PINECONE_ENV from .env
                    )
                )
            
            # Get index instance
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
                
                # Clean metadata - remove null values and ensure string types
                metadata = metadatas[i] if metadatas else {}
                cleaned_metadata = {
                    'text': text  # Always include text
                }
                
                # Add other metadata fields, ensuring no null values
                for key, value in metadata.items():
                    if value is not None:  # Only add non-null values
                        # Convert all values to strings for consistency
                        if isinstance(value, (int, float)):
                            cleaned_metadata[key] = str(value)
                        elif isinstance(value, bool):
                            cleaned_metadata[key] = str(value).lower()
                        elif isinstance(value, list):
                            cleaned_metadata[key] = [str(v) for v in value if v is not None]
                        elif isinstance(value, str):
                            cleaned_metadata[key] = value
                        else:
                            cleaned_metadata[key] = str(value)
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': cleaned_metadata
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