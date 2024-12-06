from typing import List, Tuple, Optional
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentDatabase:
    def __init__(self, vector_config):
        self.config = vector_config
        self.embeddings = None
        self.regular_vectors = None
        self.toc_vectors = None
    
    def initialize_embeddings(self):
        """Initialize embeddings with the configured model"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model
            )
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def get_document_splitters(self) -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
        common_separators = [
            "\n\nArticle",
            "\n\nCHAPTER",
            "\n\nPART",
            "\n\nSection",
            "\n\nAppendix",
            "\n\nAnnex",
            "\n\nSchedule",
            "\nPage",
            ".\n",  # Sentence breaks at new lines
            "\n\n",
            "\n",
            ". ",
            " "
        ]
        
        regular_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.regular_chunk_size,
            chunk_overlap=self.config.regular_chunk_overlap,
            separators=common_separators,
            length_function=len,
            add_start_index=True
        )
        
        toc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.toc_chunk_size,
            chunk_overlap=self.config.toc_chunk_overlap,
            separators=["\n\n", "\n", " "],
            length_function=len,
            add_start_index=True
        )
        
        return regular_splitter, toc_splitter
    
    def process_documents(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        regular_splitter, toc_splitter = self.get_document_splitters()
        
        # Process regular documents with metadata preservation
        regular_docs = regular_splitter.split_documents(docs)
        for doc in regular_docs:
            doc.metadata['content_type'] = 'regular'
        
        # Process TOC documents
        toc_docs = toc_splitter.split_documents(docs)
        for doc in toc_docs:
            doc.metadata['content_type'] = 'toc'
        
        return regular_docs, toc_docs
    
    def initialize_vector_stores(self) -> bool:
        try:
            self.initialize_embeddings()
            loader = PyPDFDirectoryLoader("./docs")
            docs = loader.load()
            
            regular_docs, toc_docs = self.process_documents(docs)
            
            self.regular_vectors = FAISS.from_documents(
                regular_docs,
                self.embeddings
            )
            
            self.toc_vectors = FAISS.from_documents(
                toc_docs,
                self.embeddings
            )
            
            return True
        except Exception as e:
            st.error(f"Error initializing vector stores: {str(e)}")
            return False
    
    def get_comprehensive_results(self, query: str, k_value: int, is_toc: bool = False) -> List[Document]:
        vectors = self.toc_vectors if is_toc else self.regular_vectors
        
        # Get results using multiple search strategies
        similarity_results = vectors.similarity_search(query, k=k_value)
        mmr_results = vectors.max_marginal_relevance_search(
            query,
            k=k_value,
            fetch_k=k_value * 3,
            lambda_mult=0.7
        )
        
        # Combine and deduplicate results
        seen_contents = set()
        combined_results = []
        
        for doc in similarity_results + mmr_results:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                combined_results.append(doc)
        
        return combined_results[:k_value * 2]
    
    def reinitialize(self):
        return self.initialize_vector_stores()