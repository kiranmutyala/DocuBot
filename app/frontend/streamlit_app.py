import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreManager
from app.backend.agent_service import AgentService

st.set_page_config(page_title="DocuBot - AI Intelligence", layout="wide")
st.title("🤖 DocuBot: Multi-Format Conversational AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "langchain_history" not in st.session_state:
    st.session_state["langchain_history"] = []

with st.sidebar:
    st.header("DocuBot Control Panel")
    uploaded_file = st.file_uploader(
        "Upload a document", 
        type=["pdf", "txt", "csv", "xlsx", "xls"]
    )

    if "doc_meta" in st.session_state:
        st.info(st.session_state["doc_meta"])

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        st.session_state["langchain_history"] = []
        st.rerun()

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state["current_file"] != uploaded_file.name:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        temp_path = f"temp{file_ext}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Indexing document & extracting metadata..."):
            processor = DocumentProcessor()
            chunks, meta_info = processor.load_and_split(temp_path)

            vector_manager = VectorStoreManager()
            st.session_state["retriever"] = vector_manager.create_store(chunks)
            st.session_state["doc_meta"] = meta_info
            st.session_state["current_file"] = uploaded_file.name
            st.session_state["messages"] = []
            st.session_state["langchain_history"] = []

        st.sidebar.success(f"Loaded: {uploaded_file.name}")

# Render active chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Conversational Input Field
if query := st.chat_input("Ask DocuBot anything about your document..."):
    if "retriever" not in st.session_state:
        st.error("Please upload a document first before asking questions.")
    else:
        st.chat_message("user").markdown(query)
        st.session_state["messages"].append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("DocuBot is thinking..."):
                agent = AgentService()
                doc_meta = st.session_state.get("doc_meta", "")
                answer = agent.answer_question(
                    st.session_state["retriever"], 
                    query, 
                    st.session_state["langchain_history"],
                    doc_meta=doc_meta
                )
                st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["langchain_history"].append(HumanMessage(content=query))
        st.session_state["langchain_history"].append(AIMessage(content=answer))
