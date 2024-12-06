def get_css_styles():
    return """
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

            .main > div {
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 1200px;
                margin: 0 auto;
            }

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
                box-shadow: 0 0 1px #ff5c75 !important;
            }

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

            h1 {
                margin-bottom: 2rem !important;
                padding-top: 1rem !important;
            }

            .input-container {
                margin-bottom: 2rem;
            }

            .keyboard-hint {
                color: #666;
                font-size: 0.8rem;
                text-align: right;
                margin-top: 4px;
            }

            .stApp {
                background-color: #1a1a1a;
            }

            .stTextArea label {
                color: white !important;
            }

            .stTextArea textarea::placeholder {
                color: rgba(255, 255, 255, 0.5) !important;
            }
        </style>
    """