# DocuBot - Multi-Format AI Agentic Document Intelligence System
An enterprise-grade, conversational document Q&A platform built on Retrieval-Augmented Generation (RAG) with safety guardrails and multi-format document support (PDF, TXT, CSV, Excel). Powered by Streamlit, LangChain, ChromaDB, and the Groq API.

## System Architecture & Workflow

1. User File Upload (PDF, TXT, CSV, Excel)
2. Document Processor: Extracts Text & Metadata Header (Pages, Rows, Sheets)
3. Text Chunking: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)
4. Vector Store: SentenceTransformers Embeddings (all-MiniLM-L6-v2) -> ChromaDB
5. DocuBot Agentic Engine: Ingests User Query + Chat History + Metadata Context
6. Groq LLM Service: Applies Safety Guardrails & Generates Grounded Response
7. Streamlit Frontend: Multi-turn Conversational Chat UI (st.chat_input)

## Agent Roles & Capabilities

- Document Parsing Agent: Dynamically parses .pdf, .txt, .csv, .xlsx, and .xls files.
- Metadata Intelligence Agent: Prepends total pages, sheet lists, and table dimensions.
- Retrieval & Context Verification Agent: Fetches relevant semantic text chunks from ChromaDB.
- Guardrail & Safety Agent: Enforces strict boundary rules ("DO NOT HALLUCINATE").

## System Setup & Execution Guide

1. Activate Virtual Environment:
   source venv/bin/activate

2. Install Dependencies:
   pip install -r requirements.txt

3. Set Up Environment Secrets:
   cp .env.example .env
   (Edit .env to add your GROQ_API_KEY)

4. Run the Application:
   streamlit run app/frontend/streamlit_app.py

## Limitations & Challenges

- In-memory vector store indexing per session.
- Pre-extracting metadata header chunks was implemented to prevent structural context loss during similarity search.
