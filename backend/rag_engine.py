from typing import List, Optional
from components.config import Config
from components.llm import LLMFactory
from components.vector_store import VectorStoreManager
from components.document_processor import DocumentProcessor
from components.rag_graph import RAGGraph
import json
import uuid
from langchain_core.messages import AIMessage
 

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

    def query(self, user_query: str, chat_history: List[dict] = [], conversation_id: str = None, username: str = "User") -> dict:
        if not self.llm or not self.rag_graph:
            return {"response": "System Error: LLM not initialized.", "sources": [], "conversation_id": conversation_id}

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            final_state = self.rag_graph.run(user_query, chat_history, username=username, thread_id=conversation_id)
            
            response_text = final_state.get("response", "")
            if not response_text and final_state.get("messages"):
                from langchain_core.messages import AIMessage
                for m in reversed(final_state["messages"]):
                    if isinstance(m, AIMessage) and m.content:
                        response_text = m.content
                        break
            
            if not final_state.get("is_valid", True) and final_state.get("retry_count", 0) >= Config.ITERATION_COUNT:
                response_text = "There is no relevant information in the given data"

            return {
                "response": response_text,
                "sources": final_state.get("sources", []),
                "conversation_id": conversation_id
            }
        except Exception as e:
            return {"response": f"Error: {str(e)}", "sources": [], "conversation_id": conversation_id}

    def clear_memory(self):
        try:
            self.vector_store_manager.delete_collection()
            return "Background knowledge cleared."
        except Exception as e:
            return f"Error clearing knowledge: {str(e)}"

    def query_stream(self, user_query: str, chat_history: List[dict] = [], conversation_id: str = None, username: str = "User"):
        
        if not self.llm or not self.rag_graph:
            yield json.dumps({"type": "error", "content": "System Error: LLM not initialized."}) + "\n"
            return
        
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            stream = self.rag_graph.run_stream(user_query, chat_history, username=username, thread_id=conversation_id)
            
            final_response = ""
            final_sources = []
            is_valid = False
            retry_count = 0
            
            for event in stream:
                for node, values in event.items():
                    if "messages" in values:
                        for m in values["messages"]:
                            if isinstance(m, AIMessage) and m.content:
                                print(f"[THINKING]: {m.content}")
                                yield json.dumps({"type": "thinking", "content": m.content}) + "\n"
                    
                    if "response" in values:
                         final_response = values["response"]
                    
                    if "sources" in values:
                         final_sources = values["sources"]
                    
                    if "is_valid" in values:
                         is_valid = values["is_valid"]
                    
                    if "retry_count" in values: 
                         retry_count = values["retry_count"]
            
            if not is_valid and retry_count >= Config.ITERATION_COUNT:
                final_response = "There is no relevant information in the given data to answer your query accurately."
            
            if not final_response:
                final_response = "I couldn't generate a response."

            print(f"[FINAL]: {final_response}")
            yield json.dumps({
                "type": "final",
                "response": final_response,
                "sources": final_sources,
                "conversation_id": conversation_id
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
