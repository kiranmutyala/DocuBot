import os
from typing import List
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """
    Document Processor for DocuBot.
    Handles extraction and chunking for PDF, TXT, CSV, and Excel (XLSX/XLS) files.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, file_path: str, original_filename: str = None) -> List[Document]:
        """
        Extracts content from supported document types and returns chunked Document objects.
        """
        display_name = original_filename or os.path.basename(file_path)
        file_ext = os.path.splitext(display_name)[1].lower()
        documents = []

        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()

        elif file_ext == ".csv":
            df = pd.read_csv(file_path)
            text_content = df.to_string(index=False)
            documents = [Document(page_content=text_content, metadata={"source": display_name})]

        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
            text_content = df.to_string(index=False)
            documents = [Document(page_content=text_content, metadata={"source": display_name})]

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Standardize metadata source field to display_name
        for doc in documents:
            doc.metadata["source"] = display_name

        # Chunk documents
        chunks = self.text_splitter.split_documents(documents)
        return chunks
