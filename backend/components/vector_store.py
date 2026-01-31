import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .config import Config

class VectorStoreManager:
    def __init__(self):
        self._embedding_function = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        try:
            print(f"Connecting to ChromaDB at {Config.CHROMA_HOST}:{Config.CHROMA_PORT}...")
            self._client = chromadb.HttpClient(host=Config.CHROMA_HOST, port=Config.CHROMA_PORT)
            self._vector_store = Chroma(
                client=self._client,
                embedding_function=self._embedding_function,
                collection_name=Config.COLLECTION_NAME
            )
            print(f"Successfully connected to collection: {Config.COLLECTION_NAME}")
        except Exception as e:
            print(f"Failed to connect to ChromaDB: {e}")
            raise e

    @property
    def vector_store(self):
        return self._vector_store

    def delete_collection(self):
        self._client.delete_collection(Config.COLLECTION_NAME)
        self._vector_store = Chroma(
            client=self._client,
            embedding_function=self._embedding_function,
            collection_name=Config.COLLECTION_NAME
        )
