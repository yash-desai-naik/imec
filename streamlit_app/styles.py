def get_css_styles():
    return """
        <style>
            /* Base styles */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}

            /* Improved text contrast */
            .stApp {
                background-color: #1a1a1a;
            }
            
            .stApp div, .stApp p, .stApp label {
                color: rgba(255, 255, 255, 0.95) !important;
                font-weight: 400;
            }
            
            /* Headers with better visibility */
            .stApp h1, .stApp h2, .stApp h3 {
                color: rgba(255, 255, 255, 1) !important;
                font-weight: 600 !important;
            }
            
            /* Enhanced text area styling */
            .stTextArea textarea {
                min-height: 100px !important;
                border-radius: 10px !important;
                padding: 12px !important;
                font-size: 16px !important;
                background-color: #2b2b2b !important;
                color: rgba(255, 255, 255, 0.95) !important;
                border: 1px solid #404040 !important;
                font-weight: 400;
                line-height: 1.5;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            
            /* Text area focus and placeholder */
            .stTextArea textarea:focus {
                border-color: #ff5c75 !important;
                box-shadow: 0 0 1px #ff5c75 !important;
            }
            
            .stTextArea textarea::placeholder {
                color: rgba(255, 255, 255, 0.6) !important;
                font-weight: 400;
            }
            
            /* Button with improved contrast */
            .stButton > button {
                width: auto !important;
                padding: 0.5rem 2rem !important;
                font-size: 16px;
                font-weight: 500;
                border-radius: 10px;
                background-color: #ff5c75;
                color: white !important;
                border: none;
                transition: all 0.3s ease;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            
            .stButton > button:hover {
                background-color: #ff3d5a;
                transform: translateY(-1px);
            }
            
            /* Markdown text enhancements */
            .stMarkdown {
                color: rgba(255, 255, 255, 0.95) !important;
                line-height: 1.6;
            }
            
            /* Expander improvements */
            .streamlit-expanderHeader {
                color: rgba(255, 255, 255, 0.95) !important;
                font-weight: 500 !important;
            }
            
            /* Source sections and quotes */
            blockquote {
                border-left: 3px solid #ff5c75;
                padding-left: 1rem;
                color: rgba(255, 255, 255, 0.9) !important;
            }
            
            /* Warning and info messages */
            .stAlert {
                background-color: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.2);
            }
            
            .stAlert > div {
                color: rgba(255, 255, 255, 0.95) !important;
            }
            
            /* Keyboard hint with better visibility */
            .keyboard-hint {
                color: rgba(255, 255, 255, 0.6) !important;
                font-size: 0.8rem;
                margin-left: 1rem;
                font-weight: 400;
            }

            /* Answer sections */
            .element-container div[data-testid="stMarkdownContainer"] > p {
                font-size: 16px !important;
                line-height: 1.6 !important;
                color: rgba(255, 255, 255, 0.95) !important;
            }

            /* Source section styling */
            .element-container div[data-testid="stExpander"] {
                background-color: rgba(43, 43, 43, 0.3);
                border-radius: 10px;
                padding: 1px 20px;
                margin: 8px 0;
            }

            /* Links with better visibility */
            a {
                color: #ff5c75 !important;
                text-decoration: none;
            }

            a:hover {
                text-decoration: underline;
            }
        </style>
    """