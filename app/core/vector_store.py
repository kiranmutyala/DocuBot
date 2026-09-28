from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self):
        # Uses a lightweight embedding model to convert text to vectors
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_store(self, chunks):
        # Store chunks in a temporary vector store
        vector_store = Chroma.from_documents(chunks, self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})
