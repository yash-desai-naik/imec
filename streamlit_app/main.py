import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import load_config
from database import DocumentDatabase
from utils import is_toc_query
from prompts import REGULAR_PROMPT, TOC_PROMPT

# Hide sidebar by default and set dark theme
st.set_page_config(
    page_title="IMEC Document Q&A System",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# Add custom CSS for improved UI
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Sidebar styling */
        [data-testid="collapsedControl"] {
            display: flex;
            justify-content: center;
            align-items: center;
            color: #0f4c81;
        }
        
        /* Container styling */
        .main > div {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            min-height: 100px !important;
            border-radius: 10px !important;
            padding: 12px !important;
            font-size: 16px !important;
            background-color: #2b2b2b !important;
            color: white !important;
            border: 1px solid #404040 !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #ff5c75 !important;
            box-shadow: 0 0 0 1px #ff5c75 !important;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            padding: 0.5rem 1rem;
            font-size: 16px;
            font-weight: 500;
            border-radius: 10px;
            margin-top: 4px;
            background-color: #ff5c75;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #ff3d5a;
            transform: translateY(-1px);
        }
        
        /* Title styling */
        h1 {
            margin-bottom: 2rem !important;
            padding-top: 1rem !important;
        }
        
        /* Input container spacing */
        .input-container {
            margin-bottom: 2rem;
        }
        
        /* Keyboard shortcut hint */
        .keyboard-hint {
            color: #666;
            font-size: 0.8rem;
            text-align: right;
            margin-top: 4px;
        }
    </style>
""", unsafe_allow_html=True)

def init_llm(api_key: str, config):
    """Initialize the LLM"""
    return ChatGroq(
        groq_api_key=api_key,
        model_name=config.model_name,
        max_tokens=config.max_tokens
    )

def process_question(question: str, llm, db, config):
    """Process a question and display results"""
    start_time = time.process_time()
    
    try:
        # Determine prompt and search parameters
        is_toc = is_toc_query(question)
        prompt = TOC_PROMPT if is_toc else REGULAR_PROMPT
        k_value = config.toc_k_value if is_toc else config.search_k_value
        
        # Get relevant documents
        with st.spinner("Searching documents..."):
            results = db.get_comprehensive_results(question, k_value, is_toc)
        
        # Create and run the chain
        with st.spinner("Analyzing content..."):
            # Create the document chain
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt
            )
            
            # Prepare input with combined document content
            chain_input = {
                "context": results,
                "input": question
            }
            
            # Get response
            response = document_chain.invoke(chain_input)
        
        processing_time = time.process_time() - start_time
        
        # Display results
        st.write("### Answer")
        if isinstance(response, dict):
            st.markdown(response.get("answer", response))
        else:
            st.markdown(response)
        
        st.write(f"_Processing time: {processing_time:.2f} seconds_")
        
        # Show sources
        with st.expander("📚 Source Sections", expanded=False):
            for i, doc in enumerate(results, 1):
                page_num = doc.metadata.get('page', 'Unknown')
                st.markdown(f"**Source {i} (Page {page_num}):**")
                st.markdown(doc.page_content)
                st.divider()
                
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        st.error("Please try rephrasing your question or contact support if the issue persists.")
        st.write(f"Debug info: {type(e).__name__}")

def handle_submit(question: str):
    """Handle question submission"""
    if question.strip():
        with st.spinner("Processing your question..."):
            process_question(question, st.session_state.llm, st.session_state.db, st.session_state.vector_config)
    else:
        st.warning("Please enter a question first.")

def create_input_interface():
    """Create the input interface with keyboard shortcuts"""
    # Initialize session state for the textarea value if not exists
    if 'textarea_value' not in st.session_state:
        st.session_state.textarea_value = ''
    
    # Create columns with better ratio
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_area(
            "Enter your question about the documents:",
            key="question_input",
            height=100,
            placeholder="Type your question here...",
            label_visibility="visible"
        )

    with col2:
        st.write("")  # Space for alignment
        ask_button = st.button(
            "Ask Question",
            key="ask_button",
            use_container_width=True,
            type="primary"
        )
        # Add keyboard shortcut hint
        st.markdown('<div class="keyboard-hint">Press ⌘/Ctrl + Enter to submit</div>', unsafe_allow_html=True)
    
    # Handle keyboard shortcuts using JavaScript
    st.markdown("""
        <script>
            document.addEventListener('keydown', function(e) {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    document.querySelector('button[kind="primary"]').click();
                }
            });
        </script>
        """, unsafe_allow_html=True)
    
    return question, ask_button

def main():
    # Load configuration
    api_keys, app_config, vector_config = load_config()
    
    # Store config in session state
    if 'vector_config' not in st.session_state:
        st.session_state.vector_config = vector_config
    
    # Initialize database if not already initialized
    if 'db' not in st.session_state:
        with st.spinner("Initializing document database..."):
            db = DocumentDatabase(vector_config)
            if db.initialize_vector_stores():
                st.session_state.db = db
                st.success("✅ Document database initialized successfully!")
                time.sleep(1)
                st.rerun()
    
    # Initialize LLM if not already initialized
    if 'llm' not in st.session_state:
        st.session_state.llm = init_llm(api_keys['GROQ_API_KEY'], app_config)
    
    # Setup interface
    st.title("IMEC Document Q&A System")
    
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Reinitialize Database"):
            st.session_state.db.reinitialize()
            st.success("Database reinitialized successfully!")
            st.rerun()
    
    # Create input interface
    question, ask_button = create_input_interface()
    
    # Handle submission
    if ask_button or (question and st.session_state.get('ctrl_enter', False)):
        handle_submit(question)
    
    # Reset ctrl_enter flag
    st.session_state['ctrl_enter'] = False

if __name__ == "__main__":
    main()