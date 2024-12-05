# PDF Chat Application

This application allows users to upload PDF documents and chat with them using natural language queries.

## Features
- PDF document upload and processing
- Interactive chat interface
- Document similarity search
- Natural language understanding

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
cd frontend
npm install
```

3. Start the backend server:
```bash
uvicorn main:app --reload
```

4. Start the frontend development server:
```bash
cd frontend
npm start
```

## Technology Stack
- Backend: FastAPI, LangChain, ChromaDB
- Frontend: React, Material-UI
- Document Processing: PyPDF, Sentence Transformers
