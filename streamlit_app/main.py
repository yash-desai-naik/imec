import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import load_config
from database import DocumentDatabase
from utils import is_toc_query
from prompts import REGULAR_PROMPT, TOC_PROMPT

# Hide sidebar by default
st.set_page_config(
    page_title="IMEC Document Q&A System",
    initial_sidebar_state="collapsed"
)

# Add custom CSS to hide Streamlit's default menu and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="collapsedControl"] {
            display: flex;
            justify-content: center;
            align-items: center;
            color: #0f4c81;
        }
    </style>
""", unsafe_allow_html=True)

def process_question(question: str, llm, db, config):
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

def init_llm(api_key: str, config):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=config.model_name,
        max_tokens=config.max_tokens
    )

def setup_interface():
    st.title("IMEC Document Q&A System")
    
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Reinitialize Database"):
            st.session_state['db'].reinitialize()
            st.success("Database reinitialized successfully!")
            st.rerun()

def main():
    # Load configuration
    api_keys, app_config, vector_config = load_config()
    
    # Initialize database if not already initialized
    if 'db' not in st.session_state:
        with st.spinner("Initializing document database..."):
            db = DocumentDatabase(vector_config)
            if db.initialize_vector_stores():
                st.session_state['db'] = db
                st.success("✅ Document database initialized successfully!")
                time.sleep(1)
                st.rerun()
    
    # Setup interface
    setup_interface()
    
    # Initialize LLM
    llm = init_llm(api_keys['GROQ_API_KEY'], app_config)
    
    # Question input and processing
    question = st.text_input("Enter your question about the documents:", key="question_input")
    
    if question and 'db' in st.session_state:
        with st.spinner("Processing your question..."):
            process_question(question, llm, st.session_state['db'], vector_config)
    elif 'db' not in st.session_state:
        st.warning("Please wait for the document database to initialize.")

if __name__ == "__main__":
    main()