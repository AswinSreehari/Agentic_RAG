import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_pdf(self, file_path: str, original_filename: str) -> str:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            for i, doc in enumerate(docs):
                doc.metadata["source"] = original_filename
                p_val = doc.metadata.get("page")
                if p_val is None:
                    p_val = doc.metadata.get("page_label") or doc.metadata.get("page_number")
                doc.metadata["page"] = p_val if p_val is not None else i

            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                return "No content extracted from PDF."

            ids = [str(uuid.uuid4()) for _ in chunks]
            self.vector_store.add_documents(documents=chunks, ids=ids)
            
            return f"Successfully processed {original_filename}."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
