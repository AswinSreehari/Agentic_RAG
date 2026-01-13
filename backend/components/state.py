from typing import List, TypedDict, Optional

class AgentState(TypedDict):
    query: str
    chat_history: List[dict]
    context: str
    response: str
    is_valid: bool
    retry_count: int
    sources: List[dict]  
    fallback_message: str
