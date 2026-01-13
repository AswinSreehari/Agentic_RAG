from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .state import AgentState
from .config import Config

class RAGGraph:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {
                "retry": "retrieve",
                "end": END
            }
        )
        
        return workflow.compile()

    def _retrieve_node(self, state: AgentState):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["query"])
        
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = []
        for doc in docs:
            sources.append({
                "filename": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
            
        print(f"\n[Retrieved Context from Sources]")
        for s in sources:
            print(f"- File: {s['filename']}, Page: {s['page']}")
            
        return {
            "context": context if context else "No context found.",
            "sources": sources
        }

    def _generate_node(self, state: AgentState):
        system_prompt = f"Answer the user query based ONLY on the provided context.\nContext:\n{state['context']}"
        messages = [SystemMessage(content=system_prompt)]
        
        for msg in state["chat_history"]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content")))
            else:
                messages.append(AIMessage(content=msg.get("content")))
        
        messages.append(HumanMessage(content=state["query"]))
        response = self.llm.invoke(messages)
        return {"response": response.content}

    def _validate_node(self, state: AgentState):
        validation_prompt = f"""
        Validate if the following response is strictly grounded in the provided context.
        Context: {state['context']}
        Response: {state['response']}
        
        Reply with exactly 'VALID' or 'INVALID'.
        """
        validation_result = self.llm.invoke([HumanMessage(content=validation_prompt)])
        is_valid = "VALID" in validation_result.content.upper()
        
        retry_count = state.get("retry_count", 0)
        
        if not is_valid:
            print(f"Validation failed. Retry count: {retry_count + 1}/{Config.ITERATION_COUNT}")
            return {"is_valid": False, "retry_count": retry_count + 1}
        
        return {"is_valid": True}

    def _should_retry(self, state: AgentState):
        if state["is_valid"]:
            return "end"
        
        if state["retry_count"] < Config.ITERATION_COUNT:
            return "retry"
        
        state["response"] = "There is no relevant information in the given data"
        return "end"

    def run(self, query: str, chat_history: list):
        initial_state = {
            "query": query,
            "chat_history": chat_history,
            "context": "",
            "response": "",
            "is_valid": False,
            "retry_count": 0,
            "sources": [],
            "fallback_message": "There is no relevant information in the given data"
        }
        return self.graph.invoke(initial_state)
