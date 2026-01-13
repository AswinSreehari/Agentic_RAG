from langchain_groq import ChatGroq
from .config import Config

class LLMFactory:
    @staticmethod
    def get_llm():
        if not Config.GROQ_API_KEY:
            return None
        return ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0.1
        )
