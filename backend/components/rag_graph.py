from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .state import AgentState
from .config import Config
import json
import re

class RAGGraph:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.graph = self._build_graph()

    def _retrieve_docs(self, query: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        sources = []
        unique_contents = set()
        context = ""
        for d in docs:
            txt = d.page_content.strip()
            if not txt or txt in unique_contents: continue
            unique_contents.add(txt)
            p = d.metadata.get("page")
            
            if p is None or not isinstance(p, int) or p <= 0:
                p = 1
                
            s = {"filename": d.metadata.get("source", "Unknown"), "page": p, "content": txt}
            sources.append(s)
            context += f"\n[File: {s['filename']}, Page: {s['page']}]\n{txt}\n"
        return {"context": context, "sources": sources}

    def _agent(self, state: AgentState):
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=(
                "You are an expert ReAct Agent. To answer, you MUST search documents first. "
                "Current Strategy: \n"
                "1. If you need info (which you usually do), output: ACTION: search_documents(\"query\")\n"
                "2. If you have sufficient info from OBSERVATIONS, output: FINAL_ANSWER: your clean response\n"
                "Constraint: FINAL_ANSWER must be clean, grounded in context, and have NO inline citations."
            )))
        res = self.llm.invoke(messages)
        return {"messages": [res]}

    def _tool_executor(self, state: AgentState):
        last_msg = state["messages"][-1].content
        match = re.search(r'ACTION:\s*search_documents\("(.*)"\)', last_msg)
        if match:
            q = match.group(1)
            res = self._retrieve_docs(q)
            obs = AIMessage(content=f"OBSERVATION: {res['context']}")
            
            current_sources = state.get("sources", [])
            seen = set((s['filename'], s['page'], s['content']) for s in current_sources)
            new_sources = []
            for s in current_sources:
                new_sources.append(s)
            
            for s in res["sources"]:
                key = (s['filename'], s['page'], s['content'])
                if key not in seen:
                    seen.add(key)
                    new_sources.append(s)

            return {
                "messages": [obs],
                "context": (state.get("context", "") + "\n" + res["context"]).strip(),
                "sources": new_sources
            }
        return {"messages": []}

    def _validator(self, state: AgentState):
        ans = state["messages"][-1].content
        if "FINAL_ANSWER:" in ans:
            clean_ans = ans.split("FINAL_ANSWER:")[-1].strip()
            v_prompt = f"Context: {state['context']}\nResponse: {clean_ans}\nReply 'VALID' or 'INVALID' only."
            v_res = self.llm.invoke([HumanMessage(content=v_prompt)])
            is_valid = "VALID" in v_res.content.upper()
            
            return {
                "is_valid": is_valid, 
                "response": clean_ans, 
                "sources": state.get("sources", []), 
                "retry_count": state.get("retry_count", 0) + (0 if is_valid else 1)
            }
        return {"is_valid": False}

    def _router(self, state: AgentState):
        last_msg = state["messages"][-1].content
        if "FINAL_ANSWER:" in last_msg: return "validate"
        if "ACTION:" in last_msg: return "action"
        return "validate" 

    def _retry_logic(self, state: AgentState):
        if state["is_valid"] or state["retry_count"] >= Config.ITERATION_COUNT: return "end"
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

    def run(self, query: str, chat_history: list):
        msgs = []
        for m in chat_history:
            if m.get("role") == "user": msgs.append(HumanMessage(content=m["content"]))
            else: msgs.append(AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=query))
        init = {"query": query, "messages": msgs, "context": "", "response": "", "is_valid": False, "retry_count": 0, "sources": []}
        final = self.graph.invoke(init)
        if not final.get("is_valid") and final.get("retry_count", 0) >= Config.ITERATION_COUNT:
            final["response"] = "There is no relevant information in the given data"
            final["sources"] = []
        return final
