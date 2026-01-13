import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from .config import Config

class VectorStoreManager:
    def __init__(self):
        self._embedding_function = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self._client = chromadb.HttpClient(host=Config.CHROMA_HOST, port=Config.CHROMA_PORT)
        self._vector_store = Chroma(
            client=self._client,
            embedding_function=self._embedding_function,
            collection_name=Config.COLLECTION_NAME
        )

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
