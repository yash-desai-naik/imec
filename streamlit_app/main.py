import streamlit as st
import time
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import load_config
from database import DocumentDatabase
from utils import is_toc_query
from prompts import REGULAR_PROMPT, TOC_PROMPT
# Import and apply CSS
from styles import get_css_styles
from feedback_store import FeedbackStore


# Hide sidebar by default and set dark theme
st.set_page_config(
    page_title="IMEC Document Q&A System",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Apply CSS (place this after st.set_page_config and before any other st elements)
st.markdown(get_css_styles(), unsafe_allow_html=True)

def init_llm(api_key: str, config):
    """Initialize the LLM"""
    return ChatGroq(
        groq_api_key=api_key,
        model_name=config.model_name,
        max_tokens=config.max_tokens
    )

def process_question(question: str, llm, db, config):
    """Process a question and display results with feedback integration"""
    start_time = time.process_time()
    
    try:
        # Check for existing feedback
        feedback = st.session_state.feedback_store.get_feedback(question)
        
        # Get model response
        is_toc = is_toc_query(question)
        prompt = TOC_PROMPT if is_toc else REGULAR_PROMPT
        k_value = config.toc_k_value if is_toc else config.search_k_value
        
        with st.spinner("Searching documents..."):
            results = db.get_comprehensive_results(question, k_value, is_toc)
            
            if not results:
                st.warning("No relevant content found. Please try a different question or make sure documents are properly loaded.")
                return
        
        with st.spinner("Analyzing content..."):
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )
            
            chain_input = {
                "context": results,
                "input": question
            }
            
            response = document_chain.invoke(chain_input)
            
            if isinstance(response, dict):
                response = response.get("answer", response)
        
        processing_time = time.process_time() - start_time
        
        # If feedback exists, show it prominently
        if feedback:
            st.warning("""
                ⚠️ **Note**: Previous feedback indicates this information needed correction:
                
                {}
                
                _This feedback was provided by a user and may need verification._
                """.format(feedback["feedback"]))
        
        # Show model response
        st.markdown(response)
        
        st.write(f"_Processing time: {processing_time:.2f} seconds_")

        # Show warning that AI generated answer
        st.markdown("<p style='font-size: 0.8em; font-style: italic; color: #888;'>AI-generated response. Please double-check.</p>", unsafe_allow_html=True)
        
        # Feedback section
        with st.expander("📝 Provide Feedback", expanded=False):
            feedback_text = st.text_area(
                "If this answer needs correction, please provide feedback:",
                key="feedback_input",
                height=100
            )
            
            if st.button("Submit Feedback"):
                if feedback_text.strip():
                    st.session_state.feedback_store.add_feedback(
                        question=question,
                        original_answer=response,
                        feedback=feedback_text
                    )
                    st.success("Thank you for your feedback! It will help improve future responses.")
                else:
                    st.error("Please enter feedback before submitting.")
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error("Please try rephrasing your question or contact support if the issue persists.")
        st.write(f"Debug info: {type(e).__name__}")


def handle_submit(question: str):
    """Handle question submission"""
    if question.strip():
        with st.spinner("Processing your question..."):
            if 'db' in st.session_state and st.session_state.db is not None:
                process_question(question, st.session_state.llm, st.session_state.db, st.session_state.vector_config)
            else:
                st.error("Document database not properly initialized. Please try reinitializing the database.")
    else:
        st.warning("Please enter a question first.")

def create_input_interface():
    """Create the input interface with keyboard shortcuts"""
    if 'textarea_value' not in st.session_state:
        st.session_state.textarea_value = ''
    
    # Text area for question input
    question = st.text_area(
        "Enter your question about the documents:",
        key="question_input",
        height=100,
        placeholder="Type your question here...",
        label_visibility="visible"
    )
    
    # Create a flex container for button and hint
    col1, col2 = st.columns([6,4])
    
    with col1:
        with st.container():
            ask_button = st.button(
                "Ask Question",
                key="ask_button",
                type="primary"
            )
    
    return question, ask_button

def find_docs_directory():
    """Find the docs directory with better path resolution"""
    # Check multiple possible locations
    possible_paths = [
        "./docs",  # Current directory
        "docs",    # Relative path
        "../docs", # Parent directory
        "streamlit_app/docs", # App directory
        os.path.join(os.path.dirname(__file__), "docs"), # File directory
        os.path.abspath("docs"), # Absolute path
    ]
    
    # Try each path and show debug info
    found_paths = []
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            found_paths.append((path, abs_path, len([f for f in os.listdir(abs_path) if f.lower().endswith('.pdf')])))
    
    return found_paths

def ensure_docs_directory(app_config):
    """Ensure the docs directory exists with better debugging"""
    docs_dir = app_config.docs_dir
    
    # Try to find docs directory in multiple locations
    found_paths = find_docs_directory()
    
    if found_paths:
        # Display found paths
        st.info("Found potential docs directories:")
        for orig_path, abs_path, pdf_count in found_paths:
            st.code(f"Path: {orig_path}\nFull path: {abs_path}\nPDF files: {pdf_count}")
        
        # Use the first path with PDFs
        pdf_paths = [p for p in found_paths if p[2] > 0]
        if pdf_paths:
            path_to_use = pdf_paths[0][1]
            app_config.docs_dir = path_to_use
            st.success(f"Using docs directory: {path_to_use} with {pdf_paths[0][2]} PDF files")
            return True
        else:
            # Found directories but no PDFs
            st.warning("Found docs directories but no PDF files in them.")
    
    # Nothing found - show current directory info
    st.error("Could not find docs directory with PDF files")
    st.info(f"Current working directory: {os.getcwd()}")
    st.info(f"Directory contents: {os.listdir('.')}")
    
    # Check if specified docs directory exists
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        st.warning(f"Created docs directory at {docs_dir}. Please add PDF documents.")
        return False
        
    # Check if directory contains PDF files
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in {docs_dir}. Please add PDF documents.")
        return False
        
    return True

def main():
    # Load configuration
    api_keys, app_config, vector_config = load_config()
    
    # Store config in session state
    if 'vector_config' not in st.session_state:
        st.session_state.vector_config = vector_config
    
    # Setup interface
    st.title("IMEC Document Q&A System")
    
    # Enhanced docs directory check
    docs_available = ensure_docs_directory(app_config)
    
    # Initialize feedback store if not already initialized
    if 'feedback_store' not in st.session_state:
        st.session_state.feedback_store = FeedbackStore()
    
    # Initialize LLM if not already initialized
    if 'llm' not in st.session_state:
        try:
            st.session_state.llm = init_llm(api_keys['GROQ_API_KEY'], app_config)
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.error("Please check your API keys in the .env file.")
            return
    
    # Handle the database initialization
    if 'db' not in st.session_state or st.session_state.db is None:
        if docs_available:
            with st.spinner("Initializing document database..."):
                try:
                    db = DocumentDatabase(vector_config)
                    # Update the docs_dir in the database to match found path
                    db.docs_dir = app_config.docs_dir
                    if db.initialize_vector_stores():
                        st.session_state.db = db
                        st.success("✅ Document database initialized successfully!")
                    else:
                        st.session_state.db = None
                        st.warning("Failed to initialize the database. Please check your documents and try again.")
                except Exception as e:
                    st.session_state.db = None
                    st.error(f"Error initializing database: {str(e)}")
                    st.error("Please check your documents and API keys.")
        else:
            st.warning("Please add PDF documents to the docs directory before initializing the database.")
            
    # Add sidebar controls
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Reinitialize Database"):
            if 'db' in st.session_state:
                # Properly clear database from session state before reinitializing
                old_db = st.session_state.db
                st.session_state.db = None
                
                # Only try to reinitialize if documents are available
                if docs_available:
                    with st.spinner("Reinitializing database..."):
                        try:
                            db = DocumentDatabase(vector_config)
                            # Update the docs_dir in the database to match found path
                            db.docs_dir = app_config.docs_dir
                            if db.initialize_vector_stores():
                                st.session_state.db = db
                                st.success("Database reinitialized successfully!")
                                st.rerun()
                            else:
                                st.warning("Failed to reinitialize the database. Please check your documents.")
                        except Exception as e:
                            st.error(f"Error reinitializing database: {str(e)}")
                else:
                    st.warning("Please add PDF documents to the docs directory first.")
    
    # Create input interface and handle submission
    question, ask_button = create_input_interface()
    
    if ask_button or (question and st.session_state.get('ctrl_enter', False)):
        handle_submit(question)
    
    st.session_state['ctrl_enter'] = False

if __name__ == "__main__":
    main()
