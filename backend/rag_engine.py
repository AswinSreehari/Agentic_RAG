from typing import List, Optional
from components.config import Config
from components.llm import LLMFactory
from components.vector_store import VectorStoreManager
from components.document_processor import DocumentProcessor
from components.rag_graph import RAGGraph

class RAGService:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.llm = LLMFactory.get_llm()
        self.doc_processor = DocumentProcessor(self.vector_store_manager.vector_store)
        
        if self.llm:
            self.rag_graph = RAGGraph(self.llm, self.vector_store_manager.vector_store)
        else:
            self.rag_graph = None

    def ingest_file(self, file_path: str, original_filename: str) -> str:
        return self.doc_processor.process_pdf(file_path, original_filename)

    def query(self, user_query: str, chat_history: List[dict] = []) -> dict:
        if not self.llm or not self.rag_graph:
            return {"response": "System Error: LLM not initialized.", "sources": []}

        try:
            final_state = self.rag_graph.run(user_query, chat_history)
            
            response_text = final_state["response"]
            if not final_state["is_valid"] and final_state["retry_count"] >= Config.ITERATION_COUNT:
                response_text = "There is no relevant information in the given data"

            return {
                "response": response_text,
                "sources": final_state["sources"]
            }
        except Exception as e:
            return {"response": f"Error: {str(e)}", "sources": []}

  
