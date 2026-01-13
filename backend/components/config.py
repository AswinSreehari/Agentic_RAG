import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    ITERATION_COUNT = int(os.getenv("ITERATION_COUNT", "3"))
    COLLECTION_NAME = "rag_docs"
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
