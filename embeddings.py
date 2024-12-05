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
