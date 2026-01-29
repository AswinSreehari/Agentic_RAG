import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    ITERATION_COUNT = int(os.getenv("ITERATION_COUNT", "3"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    MODEL_NAME = os.getenv("MODEL_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
