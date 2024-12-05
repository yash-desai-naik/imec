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
            raise