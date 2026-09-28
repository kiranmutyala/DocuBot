import os
import tempfile
import streamlit as st
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore
from app.backend.agent_service import AgentService

st.set_page_config(
    page_title="DocuBot - Multi-Format Conversational RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DocuBot: Multi-Format Conversational RAG")
st.caption("Upload PDF, TXT, CSV, or Excel documents and ask natural language questions.")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "agent_service" not in st.session_state:
    st.session_state.agent_service = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Document Ingestion & Reset Controls
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Enterprise Documents",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if st.button("Process & Embed Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing and chunking documents..."):
                # Clear chat history and initialize clean vector store
                st.session_state.messages = []
                vs = VectorStore()
                vs.reset()  # Wipe existing embeddings from disk

                processor = DocumentProcessor()
                all_chunks = []

                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    try:
                        chunks = processor.process_file(tmp_file_path, original_filename=uploaded_file.name)
                        all_chunks.extend(chunks)
                    finally:
                        if os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)

                if all_chunks:
                    vs.add_documents(all_chunks)
                    st.session_state.vector_store = vs
                    st.session_state.agent_service = AgentService(vector_store=vs)
                    st.success(f"Successfully processed {len(all_chunks)} text chunks!")
                else:
                    st.error("No valid text content extracted from uploaded files.")

    st.divider()
    if st.button("Reset Conversation & Session"):
        if st.session_state.vector_store is not None:
            st.session_state.vector_store.reset()
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.agent_service = None
        st.rerun()

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Referenced Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# Chat input prompt
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.agent_service is None:
            fallback_ans = "Please upload and process documents in the sidebar before asking questions."
            st.markdown(fallback_ans)
            st.session_state.messages.append({"role": "assistant", "content": fallback_ans})
        else:
            with st.spinner("Analyzing query and searching documents..."):
                response = st.session_state.agent_service.run(user_query)
                answer_text = response["answer"]
                sources = response.get("sources", [])

                st.markdown(answer_text)
                if sources:
                    with st.expander("📚 Referenced Sources"):
                        for src in sources:
                            st.markdown(f"- `{src}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
