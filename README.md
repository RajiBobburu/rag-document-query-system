# RAG Document Query System
An AI-powered assistant that enables users to query documents and web content using Retrieval-Augmented Generation (RAG).
--------------------------------------
## Overview

This project implements a **RAG (Retrieval-Augmented Generation)** system that allows users to:
- Upload documents or provide website URLs  
- Ask questions in natural language  
- Receive accurate, context-aware answers  
The system uses **FAISS for vector search** and **Google Gemini for response generation**.
--------------------------------------
## Key Features
- Semantic search using FAISS vector database  
- Website content ingestion  
- Document upload support  
- Multi-turn conversational chat  
- Persistent storage (vector database reuse)  
- Download chat history  
- Fast and efficient retrieval-based responses  
- Context-aware answer generation  
--------------------------------------
## Architecture
- User Input
↓
Data Ingestion (File / URL)
↓
Text Chunking
↓
Embeddings Generation
↓
FAISS Vector Store
↓
User Query
↓
Similarity Search
↓
Context Retrieval
↓
LLM (Gemini)
↓
Final Response
--------------------------------------
## Tech Stack
- **Python**
- **Streamlit** (UI)
- **FAISS** (Vector Database)
- **LangChain** (RAG pipeline)
- **Google Gemini API** (LLM)
- **BeautifulSoup** (Web scraping)
- **Sentence Transformers** (Embeddings)
--------------------------------------
## How to Run
1️⃣ Clone Repository
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Add API Key
- Create .env file:
```bash
GOOGLE_API_KEY=your_api_key_here
```
4️⃣ Run Application
```bash
streamlit run app.py
```
--------------------------------------
Sample Test URLs
- https://en.wikipedia.org/wiki/Call_center
- https://en.wikipedia.org/wiki/Customer_service
--------------------------------------
How It Works
- Input data is split into smaller chunks
- Each chunk is converted into embeddings
- Embeddings are stored in FAISS
- User query is converted into embedding
- FAISS retrieves similar chunks
- Context is sent to Gemini
- Gemini generates final answer
--------------------------------------
Use Cases
- Customer support automation
- Knowledge base assistants
- Document search systems
- Call center analytics support
- Internal company documentation query
--------------------------------------
Limitations
- Works only with provided data (no internet knowledge)
- Website scraping may fail for dynamic pages
- Accuracy depends on input data quality

Author : Bobburu Raji
