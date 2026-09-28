import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def load_and_split(self, file_path: str):
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        documents = []
        meta_info = ""

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            page_count = len(documents)
            meta_info = f"DOCUMENT METADATA: File Name: {file_name}, Total Pages: {page_count}, File Type: PDF."

        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            meta_info = f"DOCUMENT METADATA: File Name: {file_name}, File Type: Plain Text."

        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
            content = df.to_string(index=False)
            documents = [Document(page_content=content, metadata={"source": file_path})]
            meta_info = f"DOCUMENT METADATA: File Name: {file_name}, File Type: CSV, Total Rows: {len(df)}, Total Columns: {len(df.columns)}."

        elif file_extension in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(file_path)
            sheet_contents = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheet_contents.append(f"--- Sheet: {sheet_name} ---\n" + df.to_string(index=False))
            
            full_text = "\n\n".join(sheet_contents)
            documents = [Document(page_content=full_text, metadata={"source": file_path})]
            meta_info = f"DOCUMENT METADATA: File Name: {file_name}, File Type: Excel, Total Sheets: {len(excel_file.sheet_names)} ({', '.join(excel_file.sheet_names)})."

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        # Prepend metadata header chunk so vector store can retrieve structural info
        meta_doc = Document(
            page_content=f"{meta_info} Total Extracted Chunks: {len(chunks)}.",
            metadata={"source": file_path}
        )
        chunks.insert(0, meta_doc)
        
        return chunks, meta_info
