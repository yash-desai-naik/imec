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
        query = re.sub(r'(?i)art\.?\s*(\d+)', r'Article \1', query)
        query = re.sub(r'(?i)article\s+(\d+)', r'Article \1', query)
        
        # Fix double word issue
        query = query.replace('explainlain', 'explain')
        
        # Remove any multiple spaces
        query = ' '.join(query.split())
        
        logger.info(f"Preprocessed query: {query}")
        return query

    def process_query(self, query: str) -> str:
        try:
            logger.info(f"Processing query: {query}")
            processed_query = self._preprocess_query(query)
            
            # Extract article number if present
            article_match = re.search(r'(?i)Article\s+(\d+)', processed_query)
            article_num = article_match.group(1) if article_match else None
            
            # Get relevant documents
            k = 4
            docs = self.vector_store.similarity_search(processed_query, k=k * 2)  # Get more docs initially
            
            # Filter and sort documents
            if article_num and docs:
                # Prioritize exact article matches
                docs = sorted(docs, key=lambda x: (
                    x['metadata'].get('article_number') == article_num,  # Exact matches first
                    x['metadata'].get('score', 0)  # Then by score
                ), reverse=True)
                docs = docs[:k]  # Take top k after sorting
            
            if not docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Create context with better structure
            context_parts = []
            for doc in docs:
                score = doc['metadata'].get('score', 0)
                if score >= 0.3:
                    article_num = doc['metadata'].get('article_number', '')
                    article_title = doc['metadata'].get('article_title', '')
                    is_partial = doc['metadata'].get('is_partial', False)
                    part_info = f"(Part {doc['metadata'].get('part_number')}/{doc['metadata'].get('total_parts')})" if is_partial else ""
                    
                    context = (
                        f"Article {article_num}: {article_title} {part_info}\n"
                        f"{'=' * 40}\n"
                        f"{doc['text'].strip()}\n"
                        f"Relevance Score: {score:.2f}\n"
                        f"{'-' * 40}"
                    )
                    context_parts.append(context)

            if not context_parts:
                return "I couldn't find any sufficiently relevant information to answer your question accurately. Could you please rephrase or be more specific?"

            prompt_template = """You are analyzing the IBF-AMOSUP/IMEC Agreement for 2024-2025. Answer the question using only the provided article(s). Be specific and detail-oriented.

            Document Context:
            {context_separator}
            {context_content}
            {context_separator}

            Question: {question}

            Instructions:
            1. Answer based ONLY on the provided article(s)
            2. Quote specific sections when relevant
            3. Include article numbers and titles in your response
            4. If information is missing or unclear, say so
            5. Use professional, precise language
            6. If asked about a specific article, focus on that article's content
            7. Format your response with:
            - Article reference (number and title)
            - Key provisions/points
            - Relevant quotes
            - Additional context if available

            Detailed Response:"""

            # Build the context content with separators
            context_separator = '-' * 80
            context_content = ('\n' + context_separator + '\n').join(context_parts)

            # Format the final prompt
            prompt = prompt_template.format(
                context_separator=context_separator,
                context_content=context_content,
                question=processed_query
            )

            print(prompt)  # For debugging


            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": """You are analyzing the IBF-AMOSUP/IMEC Agreement for 2024-2025.
                            You are a maritime legal expert who:
                            - Provides precise, article-specific responses
                            - Uses direct quotes from the text
                            - Maintains professional language
                            - Clearly indicates when information is missing
                            - Focuses on the specific article(s) mentioned in the query
                            - Structures responses in a clear, logical manner"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.1,  # Very low temperature for consistent, focused responses
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