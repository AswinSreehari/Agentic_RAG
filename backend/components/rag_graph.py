from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .state import AgentState
from .config import Config
import re

class RAGGraph:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.graph = self._build_graph()

    def _retrieve_docs(self, query: str):
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
            docs = retriever.invoke(query)
            
            grouped_docs = {}
            for d in docs:
                source = d.metadata.get("source") or d.metadata.get("doc_name") or "Unknown"
                if source not in grouped_docs:
                    grouped_docs[source] = []
                grouped_docs[source].append(d)
                
            selected_docs = []
            max_docs_per_source = max(len(v) for v in grouped_docs.values()) if grouped_docs else 0
            
            for i in range(max_docs_per_source):
                for source in grouped_docs:
                    if i < len(grouped_docs[source]):
                        selected_docs.append(grouped_docs[source][i])
                        
            selected_docs = selected_docs[:5]

            sources = []
            unique_contents = set()
            context = ""
            
            for d in selected_docs:
                txt = d.page_content.strip()
                if not txt or txt in unique_contents: continue
                unique_contents.add(txt)
                
                p = d.metadata.get("page") or d.metadata.get("page_no") or 1
                filename = d.metadata.get("source") or d.metadata.get("doc_name") or "Unknown"
                
                s = {
                    "filename": filename, 
                    "page": p, 
                    "content": txt,
                    "chunk_id": d.metadata.get("chunk_id"),
                    "chunk_index": d.metadata.get("chunk_index")
                }
                sources.append(s)
                context += f"\n[File: {s['filename']}, Page: {s['page']}]\n{txt}\n"
            return {"context": context, "sources": sources}
        except Exception as e:
            return {"context": "", "sources": []}

    def _agent(self, state: AgentState):
        messages = list(state["messages"])
        username = state.get("username", "User")
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=(
                f"You are an expert ReAct Agent assisting {username}. To answer, you MUST search documents first. "
                "Current Strategy: \n"
                "1. If you need info, output: ACTION: search_documents(\"query\")\n"
                "2. If you have sufficient info, output: FINAL_ANSWER: your response\n"
                "Constraint: FINAL_ANSWER must be grounded in context."
            )))
        res = self.llm.invoke(messages)
        return {"messages": [res], "steps": state.get("steps", 0) + 1}

    def _tool_executor(self, state: AgentState):
        if state.get("steps", 0) > 10: 
             return {"messages": [SystemMessage(content="Limit reached. Output FINAL_ANSWER now.")]}
             
        last_msg = state["messages"][-1].content
        match = re.search(r'ACTION:\s*search_documents\([\'"](.*?)[\'"]\)', last_msg, re.IGNORECASE)
        
        if match:
            q = match.group(1)
            res = self._retrieve_docs(q)
            obs_content = f"OBSERVATION: {res['context']}" if res['context'] else "OBSERVATION: No relevant documents found."
            
            obs = AIMessage(content=obs_content)
            
            current_sources = state.get("sources", [])
            seen = set((s['filename'], s['page'], s['content']) for s in current_sources)
            new_sources = list(current_sources)
            
            for s in res["sources"]:
                key = (s['filename'], s['page'], s['content'])
                if key not in seen:
                    seen.add(key)
                    new_sources.append(s)

            return {
                "messages": [obs],
                "context": (state.get("context", "") + "\n" + obs_content).strip(),
                "sources": new_sources
            }
        
        return {"messages": [AIMessage(content="OBSERVATION: Invalid format. Use ACTION: search_documents(\"query\")")]}

    def _validator(self, state: AgentState):
        ans = state["messages"][-1].content
        
        clean_ans = None
        if "FINAL_ANSWER:" in ans.upper():
            parts = re.split(r'FINAL_ANSWER:', ans, flags=re.IGNORECASE)
            clean_ans = parts[-1].strip()
        elif state.get("steps", 0) > 5:
            clean_ans = ans.strip()

        if clean_ans:
            v_prompt = f"Context: {state['context']}\nResponse: {clean_ans}\nReply 'VALID' or 'INVALID' only."
            try:
                v_res = self.llm.invoke([HumanMessage(content=v_prompt)])
                is_valid = "VALID" in v_res.content.upper()
            except:
                is_valid = True
            
            return {
                "is_valid": is_valid, 
                "response": clean_ans, 
                "sources": state.get("sources", []), 
                "retry_count": state.get("retry_count", 0) + (0 if is_valid else 1)
            }
        
        return {
            "is_valid": False,
            "retry_count": state.get("retry_count", 0) + 1
        }

    def _router(self, state: AgentState):
        last_msg = state["messages"][-1].content
        if "FINAL_ANSWER:" in last_msg.upper(): return "validate"
        if "ACTION:" in last_msg.upper(): return "action"
        if state.get("steps", 0) > 5: return "validate"
        return "validate" 

    def _retry_logic(self, state: AgentState):
        if state.get("is_valid") or state.get("retry_count", 0) >= Config.ITERATION_COUNT: 
            return "end"
        return "agent"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent)
        workflow.add_node("action", self._tool_executor)
        workflow.add_node("validate", self._validator)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._router, {"action": "action", "validate": "validate"})
        workflow.add_edge("action", "agent")
        workflow.add_conditional_edges("validate", self._retry_logic, {"agent": "agent", "end": END})
        return workflow.compile()
 
    def run(self, query: str, chat_history: list, username: str = "User", thread_id: str = "default"):
        msgs = []
        for m in chat_history:
            if m.get("role") == "user": msgs.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant": msgs.append(AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=query))
        
        init = {
            "query": query, "messages": msgs, "context": "", "response": "", 
            "is_valid": False, "retry_count": 0, "sources": [], "username": username, "steps": 0
        }
        return self.graph.invoke(init)

    def run_stream(self, query: str, chat_history: list, username: str = "User", thread_id: str = "default"):
        msgs = []
        for m in chat_history:
            if m.get("role") == "user": msgs.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant": msgs.append(AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=query))
        
        init = {
            "query": query, "messages": msgs, "context": "", "response": "", 
            "is_valid": False, "retry_count": 0, "sources": [], "username": username, "steps": 0
        }
        return self.graph.stream(init)
