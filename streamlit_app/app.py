import streamlit as st
import os
import time
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Configure API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GROQ_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Missing required API keys in .env file")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Streamlit interface
st.title("IMEC Document Q&A System")

# Add sidebar for settings
with st.sidebar:
    st.header("Settings")
    if st.button("🔄 Reinitialize Database"):
        for key in ['vectors', 'toc_vectors', 'regular_vectors', 'embeddings', 'toc_documents']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Database reinitialization triggered!")

# Define the prompt template with proper input variables
REGULAR_PROMPT = """
You are a precise document analyzer with access to multiple document chunks that may contain relevant information. Your task is to:

1. THOROUGHLY scan all provided chunks for the complete article content, including content that might continue on different pages
2. ALWAYS provide the FULL content of articles and their sections, never summarize
3. Ensure you capture ALL sections belonging to an article, even if they appear in different chunks
4. Include any referenced appendices or annexes

If asked about a specific article:
1. First, look for the article across ALL chunks
2. Then, scan ALL chunks for sections belonging to that article
3. Finally, look for any referenced appendices/annexes

Structure your response as:

Article [Number]: [Title] - Page [X]
[COMPLETE article content - Do not omit any part]

Sections:
- Section [Number]: [Title] - Page [Y]
  [COMPLETE section content]
  [If section continues on next page, include "[Continued on Page Z]" and include the continuation]

Referenced Materials:
[Only include if explicitly referenced in the article or sections]
- Appendix [Number]: [Title] - Page [W]
  [Relevant content]
- Annex [Number]: [Title] - Page [V]
  [Relevant content]

Key requirements:
- NEVER say an article doesn't exist without checking ALL chunks
- If you find partial content, indicate "Content continues..." and look for the continuation
- Always include page numbers for every part
- Maintain the exact text as written in the document
- If a section truly doesn't exist, say "This article contains no sections in the document"

Context: {context}
Question: {input}
"""

TOC_PROMPT = """
You are tasked with presenting the table of contents in a clear, structured format.

Present the TOC as:
PART [Number]: [Title]
  CHAPTER [Number]: [Title]
    Article [Number]: [Title] - Page [X]
      Section [Number]: [Title]

Keep the structure hierarchical and include page numbers.

Context: {context}
Question: {input}
"""


def init_vector_store():
    """Initialize or return existing vector stores"""
    if not all(key in st.session_state for key in ['regular_vectors', 'toc_vectors']):
        try:
            with st.spinner("Initializing document database..."):
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
                
                loader = PyPDFDirectoryLoader("./docs")
                docs = loader.load()
                
                # Enhanced article and section splitter
                regular_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=6000,     # Increased for better context
                    chunk_overlap=2000,   # Larger overlap to catch cross-page content
                    separators=[
                        "\n\nArticle",    # Primary document structure
                        "\n\nCHAPTER",
                        "\n\nPART",
                        "\n\nSection",
                        "\n\nAppendix",
                        "\n\nAnnex",
                        "\nPage",         # Page markers
                        "\n\n",           # Paragraphs
                        "\n",             # Lines
                        ". ",             # Sentences
                        " ",              # Words
                        ""
                    ],
                    length_function=len,
                    add_start_index=True,
                )
                
                # Special splitter for TOC
                toc_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=8000,      # Larger chunks for TOC
                    chunk_overlap=2000,    # Large overlap for TOC continuity
                    separators=["\n\n", "\n", " "],
                    length_function=len,
                    add_start_index=True,
                )
                
                # Process documents
                regular_documents = regular_splitter.split_documents(docs)
                st.session_state.toc_documents = toc_splitter.split_documents(docs)
                
                # Create vector stores
                st.session_state.regular_vectors = FAISS.from_documents(
                    regular_documents,
                    st.session_state.embeddings
                )
                
                st.session_state.toc_vectors = FAISS.from_documents(
                    st.session_state.toc_documents,
                    st.session_state.embeddings
                )
                
            st.success("✅ Document database initialized successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error initializing vector stores: {str(e)}")
            return False
    return True


def find_references(text):
    """Extract references to appendices and annexes"""
    references = []
    
    patterns = [
        r'appendix\s+([A-Za-z0-9]+)',
        r'annex\s+([A-Za-z0-9]+)',
        r'schedule\s+([A-Za-z0-9]+)',
        r'attachment\s+([A-Za-z0-9]+)',
        r'exhibit\s+([A-Za-z0-9]+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower())
        references.extend([match.group(1) for match in matches])
    
    return list(set(references))


def is_toc_query(query):
    """Check if the query is about table of contents"""
    toc_keywords = ['table of contents', 'toc', 'contents page', 'index', 'list of contents']
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in toc_keywords)



def get_related_content(text, vectors):
    """Find related content including appendices and annexes"""
    references = find_references(text)
    related_chunks = []
    
    if references:
        for ref in references:
            search_terms = [
                f"appendix {ref}",
                f"annex {ref}",
                f"schedule {ref}",
                f"attachment {ref}",
                f"exhibit {ref}"
            ]
            
            for term in search_terms:
                results = vectors.similarity_search(term, k=3)
                related_chunks.extend(results)
    
    # Remove duplicates while preserving order
    seen = set()
    deduped_chunks = []
    for chunk in related_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            deduped_chunks.append(chunk)
    
    return deduped_chunks

def get_page_number(text):
    """Extract page number from text if available"""
    page_match = re.search(r'Page (\d+)', text)
    return int(page_match.group(1)) if page_match else None

@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="Llama3-8b-8192",
        max_tokens=8192
    )



# Initialize LLM
llm = init_llm()

# Auto-initialize on startup
init_vector_store()

# Main interface
st.write("Ask questions about your documents. The database will automatically initialize on startup.")

# Question input
question = st.text_input("Enter your question about the documents:", key="question_input")

def get_combined_results(vectors, question, k_value=6):
    """Get comprehensive search results combining different search strategies"""
    # Extract article number if present
    article_match = re.search(r'article\s*(\d+)', question.lower())
    article_num = article_match.group(1) if article_match else None
    
    # First search: Direct article search if article number is present
    if article_num:
        direct_results = vectors.similarity_search(
            f"Article {article_num}",
            k=k_value
        )
    else:
        direct_results = []
    
    # Second search: MMR search for diversity
    mmr_results = vectors.max_marginal_relevance_search(
        question,
        k=k_value,
        fetch_k=k_value * 3,
        lambda_mult=0.7
    )
    
    # Third search: Similarity search with broader context
    similarity_results = vectors.similarity_search(
        question,
        k=k_value
    )
    
    # Combine all results while removing duplicates
    seen_contents = set()
    combined_results = []
    
    for doc in direct_results + mmr_results + similarity_results:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            combined_results.append(doc)
    
    return combined_results[:k_value * 2]  # Return top results while keeping sufficient context

# Modified question processing
if question and hasattr(st.session_state, 'regular_vectors'):
    try:
        with st.spinner("Processing your question..."):
            start_time = time.process_time()
            
            if is_toc_query(question):
                prompt = ChatPromptTemplate.from_template(TOC_PROMPT)
                vectors = st.session_state.toc_vectors
                k_value = 10
            else:
                prompt = ChatPromptTemplate.from_template(REGULAR_PROMPT)
                vectors = st.session_state.regular_vectors
                k_value = 6
            
            # Get comprehensive results
            combined_results = get_combined_results(vectors, question, k_value)
            
            # Create and invoke chain without document_variable_name
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )
            
            # Invoke chain with proper input
            response = document_chain.invoke({
                "context": combined_results,
                "input": question
            })
            
            processing_time = time.process_time() - start_time
            
            # Display results
            st.write("### Answer")
            if isinstance(response, dict) and "answer" in response:
                st.write(response["answer"])
            else:
                st.write(response)  # Changed to display direct response if not in dict form
            st.write(f"_Processing time: {processing_time:.2f} seconds_")
            
            # Show sources if needed
            with st.expander("📚 Source Sections", expanded=False):
                for i, doc in enumerate(combined_results, 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(doc.page_content)
                    st.divider()
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.warning("Please initialize the document database first.")