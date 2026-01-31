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
                
                page_meta = doc.metadata.get("page")
                if page_meta is None:
                    page_meta = doc.metadata.get("page_label") or doc.metadata.get("page_number")
                
                if page_meta is not None:
                    try:
                        doc.metadata["page"] = int(page_meta) + 1
                    except:
                        doc.metadata["page"] = i + 1
                else:
                    doc.metadata["page"] = i + 1

            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                return "No content extracted from PDF."

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
                chunk.metadata["chunk_index"] = i
                
                # Ensure source and page are present (they should be already)
                if "source" not in chunk.metadata:
                    chunk.metadata["source"] = original_filename
                
                # If page is missing (unlikely), default to 0
                if "page" not in chunk.metadata:
                    chunk.metadata["page"] = 0

            ids = [chunk.metadata["chunk_id"] for chunk in chunks]
            self.vector_store.add_documents(documents=chunks, ids=ids)
            return f"Successfully processed {original_filename}."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
