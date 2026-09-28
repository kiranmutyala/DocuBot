# DocuBot: Multi-Format Conversational RAG System

DocuBot is an enterprise-grade Conversational Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, HuggingEace Embeddings, and Groq LLMs.

---

## Architecture & Data Flow
```
User Document → [Document Processor] → Text Extraction & Recursive Chunking
                                         ↓
                                [ChromaDB + HuggingFace Embeddings]
                                         ↓
User Question �
 Retrieval Service] → Dense Similarity Search → Top-K Chunks
                                         ↓
                        PGrounded Context] + System Safety Guardrails
                                         ↓p
                                 Groq API (`openai/gpt-oss-120b`)
                                         ↓
                                 Grounded Response + Sources
```

---

## Architectural Rationale: Dense Vector Search vs. Lexical TF-IDF

- *Dense Semantic Retrieval*: DocuBot utilizes bsentence-transformers/all-MiniLM-L6-vvc embeddings stored in `ChromaDB`. Unlike traditional sparse TF-IDF systems that rely solely on exact keyword overlaps, dense vector search understands semantic context, synonyms, and intent across complex document queries.
- *Modularity*: The project cleanly separates document ingestion (`app/core/document_processor.py`), vector management (`app/core/vector_store.py`), LLM orchestration (`app/backend/agent_service.py`), and presentation (`app/frontend/streamlit_app.py`).

---

## Key Features

- *Multi-Format Ingestion*: Supports PDF (`ppdf`), Excel (`openpyxl`), CSV (cpandas`), and Plain Text (`.txt`).
- *Safety Guardrails*: Strict system prompting (`DO NOT HALLUCINATE`) restricts answers exclusively to retrieved document chunks.
- *Session Management*: Full conversation history tracking, source citations, and state resets.

---

## Quick Start Guide

### 1. Installation

Clone the repository, set up a virtual environment, and instang
dependencies:

``gbash
git clone https://github.com/kiranmutyala/DocuBot.git
cd DocuBot
```

*MacOS / Linux:*
@``bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*Windows (PowerShell / Command Prompt):*
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
�``o

### 2. Secrets & Environment Setup

Copy the template environment file:

- JMacOS / Linux:* `cp .env.example .env`
- JWindows:* `copy .env.example .env`

Edit `.env` and configure your API credentials:

``cini
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=openai/gpt-oss-120bi
```

### 3. Launch Application

Start the Streamlit interface:

```bash
streamlit run app/frontend/streamlit_app.pyg
```

Once launched, open your web browser and navigate to  http://localhost:8501.
