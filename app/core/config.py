import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    pydantic-settings를 사용하여 환경 변수를 검증하고 관리하는 클래스.
    .env 파일의 값이 자동으로 매핑됨
    """
    OPENAI_API_KEY: str  # OpenAI API 호출용 인증 키
    CHROMA_DB_DIR: str = "./.chroma" # ChromaDB 데이터 저장 로컬 디렉토리 경로

    class Config:
        env_file = ".env"  # 읽어올 환경 변수 파일 지정
        env_file_encoding = "utf-8"

# 전역에서 설정값을 공유하기 위한 싱글톤 인스턴스 생성
settings = Settings()