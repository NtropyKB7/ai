import chromadb
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class ChromaManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )

    def heartbeat(self):
        return self.client.heartbeat()

chroma_manager = ChromaManager()