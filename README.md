# DocuBot: Multi-Format Conversational RAG System

DocuBot is an enterprise-grade Conversational Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, HuggingFace Embeddings, and Groq LLMs.

---

## Architecture & Data Flow
User Document → [Document Processor] → Text Extraction & Recursive Chunking
↓
[ChromaDB + HuggingFace Embeddings]
↓
User Question → [Intent Planner Agent] → Query Rewriting / Intent Classification
↓
[Retrieval Agent] → Dense Similarity Search
↓
[Grounded Context] + System Safety Guardrails
↓
Groq API (openai/gpt-oss-120b)
↓
Grounded Response + Sources

---

## Architectural Rationale: Dense Vector Search vs. Lexical TF-IDF

- **Dense Semantic Retrieval**: DocuBot utilizes `sentence-transformers/all-MiniLM-L6-v2` embeddings stored in `ChromaDB`. Unlike traditional sparse TF-IDF systems that rely solely on exact keyword overlaps, dense vector search understands semantic context, synonyms, and intent across complex document queries.
- **Modularity**: The project cleanly separates document ingestion (`app/core/document_processor.py`), vector management (`app/core/vector_store.py`), LLM orchestration (`app/backend/agent_service.py`), and presentation (`app/frontend/streamlit_app.py`).

---

## Agentic Workflow & Agent Roles

DocuBot uses a multi-step reasoning workflow managed by `AgentService` (`app/backend/agent_service.py`):

1. **Planner / Input Inspector**: Analyzes user queries using LLM intent classification to distinguish between document-wide summary requests (`SUMMARY`) and point-lookup queries (`SPECIFIC`). Rewrites search terms automatically for broad document overviews.
2. **Retrieval Agent**: Invokes vector search tools over ChromaDB to fetch relevant document contexts with automated fallbacks to dense search if relevance distance scores are uncalibrated.
3. **Reasoning & Synthesis Engine**: Combines retrieved context with conversation history to construct grounded answers using Groq (`openai/gpt-oss-120b`).
4. **Validation Guardrail**: Ensures responses strictly reference retrieved source chunks and triggers fallbacks (`I cannot find relevant information in the uploaded documents`) when context is insufficient.

---

## Key Features

- **Multi-Format Ingestion**: Supports PDF (`pypdf`), Excel (`openpyxl`), CSV (`pandas`), and Plain Text (`.txt`).
- **Safety Guardrails**: Strict system prompting (`DO NOT HALLUCINATE`) restricts answers exclusively to retrieved document chunks.
- **Session & Database Hygiene**: Full conversation history tracking, expandable source citations, explicit batch processing, and automatic vector database resets (`reset()`) between upload sessions to eliminate stale document citations.

---

## Quick Start Guide & Local Setup

### 1. Installation

Clone the repository, set up a virtual environment, and install dependencies:

```bash
git clone https://github.com/kiranmutyala/DocuBot.git
cd DocuBot

macOS / Linux:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Windows (PowerShell / Command Prompt):
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

2. Secrets & Environment Setup
Copy the template environment file:

macOS / Linux: cp .env.example .env

Windows: copy .env.example .env

Edit .env and configure your API credentials:
GROQ_API_KEY=gsk_your_actual_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

3. Launch Application
Start the Streamlit interface:
streamlit run app/frontend/streamlit_app.py

Once launched, open your web browser and navigate to http://localhost:8501.

Production Deployment Steps
Deploying to Streamlit Community Cloud
Push your latest code to GitHub (main branch).

Go to share.streamlit.io and connect your GitHub repository.

Set the Main file path to app/frontend/streamlit_app.py.

In Advanced Settings / Secrets, paste your environment variables:
GROQ_API_KEY = "gsk_your_actual_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"

Click Deploy.

System Limitations & Development Challenges
Limitations
In-Memory / Local Vector Store: The current ChromaDB configuration runs locally. For enterprise scale, migration to managed vector stores (e.g., Pinecone or Qdrant) is recommended.

Context Window Boundaries: Very large tabular datasets (CSV/Excel) require strategic chunking to avoid exceeding LLM context limits during prompt construction.

Challenges Faced & Mitigations
Hallucination Suppression: Addressed by enforcing strict grounding system prompts (temperature=0) and explicit fallback responses when cosine similarity thresholds are not met.

Stale Vector State: Resolved by implementing explicit database reset routines (vs.reset()) during new ingestion runs to clear persistent disk collections.

Cross-Platform Compatibility: Resolved OS-specific virtual environment activation paths and file encoding discrepancies during document ingestion.
