import os
from typing import List, Any
from app.core.config import Config

# Handle Chroma import compatibility across LangChain versions
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Handle HuggingFace Embeddings import compatibility
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """
    ChromaDB Vector Store wrapper for DocuBot.
    Handles embedding generation using HuggingFace sentence-transformers
    and provides similarity search interfaces.
    """

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.store = Chroma(
            collection_name="docubot_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Any]):
        """Ingests processed document chunks into ChromaDB."""
        if documents:
            self.store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Any]:
        """Performs dense similarity search and returns top-k documents."""
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 3):
        """Performs dense similarity search and returns (document, score) tuples."""
        try:
            return self.store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            docs = self.store.similarity_search(query, k=k)
            return [(doc, 1.0) for doc in docs]

    def reset(self):
        """Clears all persisted documents from ChromaDB collection."""
        try:
            self.store.delete_collection()
            self.store = Chroma(
                collection_name="docubot_collection",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception:
            pass

    def as_retriever(self, search_kwargs: dict = None):
        """Exposes retriever interface for LangChain components."""
        kwargs = search_kwargs or {"k": 3}
        return self.store.as_retriever(search_kwargs=kwargs)
