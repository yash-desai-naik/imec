import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class VectorStoreConfig:
    regular_chunk_size: int = 6000
    regular_chunk_overlap: int = 2000
    toc_chunk_size: int = 8000
    toc_chunk_overlap: int = 2000
    search_k_value: int = 6
    toc_k_value: int = 10
    embedding_model: str = "models/embedding-001"  # Added embedding_model here

@dataclass
class AppConfig:
    title: str = "IMEC Document Q&A System"
    model_name: str = "Llama3-8b-8192"
    max_tokens: int = 8192
    docs_dir: str = "./docs"

def load_config():
    load_dotenv()
    
    api_keys = {
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    if not all(api_keys.values()):
        raise ValueError("Missing required API keys in .env file")
    
    os.environ["GOOGLE_API_KEY"] = api_keys['GOOGLE_API_KEY']
    
    return api_keys, AppConfig(), VectorStoreConfig()