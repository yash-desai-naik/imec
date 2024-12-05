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
            raise