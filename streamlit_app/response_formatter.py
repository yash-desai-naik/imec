from typing import Dict, Any
import streamlit as st

class ResponseFormatter:
    @staticmethod
    def format_display(response: str, sources: list):
        """Format and display the response with clear separation of smart answer and sources"""
        
        # Split response into smart answer and document reference if available
        parts = response.split("DOCUMENT REFERENCE:", 1)
        
        # Display smart answer
        if len(parts) > 1:
            smart_answer = parts[0].replace("SMART ANSWER:", "").strip()
            doc_reference = parts[1].strip()
        else:
            smart_answer = response
            doc_reference = ""
        
        # Display the smart answer prominently
        st.markdown(f"### 💡 Quick Answer\n{smart_answer}")
        
        # Display document references if available
        if doc_reference:
            with st.expander("📚 Detailed Information from Documents", expanded=False):
                st.markdown(doc_reference)
        
        # Display sources
        with st.expander("🔍 Source Sections", expanded=False):
            for i, doc in enumerate(sources, 1):
                page_num = doc.metadata.get('page', 'Unknown')
                st.markdown(f"**Source {i} (Page {page_num}):**")
                st.markdown(doc.page_content)
                st.divider()
    
    @staticmethod
    def format_feedback_warning(feedback: Dict[str, Any]):
        """Format feedback warning message"""
        if feedback:
            st.warning("""
                ⚠️ **Note**: Previous feedback indicates a correction:
                
                {}
                
                _This feedback was provided by a user and may need verification._
                """.format(feedback["feedback"]))