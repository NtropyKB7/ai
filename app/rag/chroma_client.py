import chromadb
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

class ChromaManager:
    """
    Chroma Vector DB 및 임베딩 모델 인스턴스를 관리하는 클래스
    """
    def __init__(self):
        # 로컬 디렉토리에 지식 베이스(Vector) 데이터를 영속적으로 저장하기 위한 PersistentClient 설정
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

        # OpenAI의 최신 임베딩 모델(text-embedding-3-small) 초기화
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )

    def heartbeat(self):
        """
        ChromaDB 커넥션 상태 체크용 핑(ping) 헬스 체크 메서드
        """
        return self.client.heartbeat()

# 전역에서 재사용할 ChromaDB 매니저 싱글톤 인스턴스 생성
chroma_manager = ChromaManager()