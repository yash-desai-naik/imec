import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import load_config
from database import DocumentDatabase
from utils import is_toc_query  # Changed from streamlit_app.utils to utils
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
        
        # Display results with feedback
        # st.write("### Answer")
        
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
        # # Show sources
        # with st.expander("📚 Source Sections", expanded=False):
        #     for i, doc in enumerate(results, 1):
        #         page_num = doc.metadata.get('page', 'Unknown')
        #         st.markdown(f"**Source {i} (Page {page_num}):**")
        #         st.markdown(doc.page_content)
        #         st.divider()
        
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
            process_question(question, st.session_state.llm, st.session_state.db, st.session_state.vector_config)
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
    # Initialize feedback store if not already initialized
    if 'feedback_store' not in st.session_state:
        st.session_state.feedback_store = FeedbackStore()
    
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
    
    # Create input interface and handle submission
    question, ask_button = create_input_interface()
    
    if ask_button or (question and st.session_state.get('ctrl_enter', False)):
        handle_submit(question)
    
    st.session_state['ctrl_enter'] = False

if __name__ == "__main__":
    main()
