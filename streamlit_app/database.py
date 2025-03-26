from typing import List, Tuple, Optional
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

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
            return True
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return False
    
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
        """Process documents with safe handling for empty document lists"""
        if not docs:
            st.warning("No documents found in the docs directory. Please add PDF documents.")
            return [], []
            
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
        """Initialize the vector stores with error handling"""
        try:
            # First initialize embeddings
            if not self.initialize_embeddings():
                return False
                
            # Check if docs directory exists
            docs_dir = "./docs"
            if not os.path.exists(docs_dir):
                os.makedirs(docs_dir)
                st.warning(f"Created empty docs directory at {docs_dir}. Please add PDF documents.")
                return False
                
            # Check if directory contains PDF files
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            if not pdf_files:
                st.warning(f"No PDF files found in {docs_dir}. Please add PDF documents.")
                return False
            
            # Load documents
            loader = PyPDFDirectoryLoader(docs_dir)
            docs = loader.load()
            
            if not docs:
                st.warning("No content could be extracted from the PDF files. Please check the documents.")
                return False
            
            # Process documents
            regular_docs, toc_docs = self.process_documents(docs)
            
            if not regular_docs or not toc_docs:
                st.warning("Could not process documents properly. Please check file content.")
                return False
                
            # Create vector stores
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
        """Get search results with error handling"""
        try:
            if is_toc and self.toc_vectors:
                vectors = self.toc_vectors
            elif not is_toc and self.regular_vectors:
                vectors = self.regular_vectors
            else:
                st.error("Vector stores not properly initialized")
                return []
            
            # Get results using multiple search strategies
            similarity_results = vectors.similarity_search(query, k=k_value)
            
            try:
                mmr_results = vectors.max_marginal_relevance_search(
                    query,
                    k=k_value,
                    fetch_k=k_value * 3,
                    lambda_mult=0.7
                )
            except Exception:
                # Fall back to similarity search if MMR fails
                mmr_results = []
            
            # Combine and deduplicate results
            seen_contents = set()
            combined_results = []
            
            for doc in similarity_results + mmr_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    combined_results.append(doc)
            
            return combined_results[:k_value * 2] if combined_results else []
            
        except Exception as e:
            st.error(f"Error retrieving search results: {str(e)}")
            return []
    
    def reinitialize(self):
        """Reinitialize vector stores"""
        self.embeddings = None
        self.regular_vectors = None
        self.toc_vectors = None
        return self.initialize_vector_stores()
