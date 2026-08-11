from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    환경 변수 및 앱 전역 설정 관리 클래스입니다.
    """

    OPENAI_API_KEY: str

    # ------------------------------------------------------------------
    # ChromaDB / 금융상품 지식 DB 관련 설정
    # ------------------------------------------------------------------
    CHROMA_DB_DIR: str = "./.chroma"
    FINANCIAL_PRODUCT_COLLECTION: str = "financial_products"

    # 현재 원천 상품 데이터 파일 경로
    FINANCIAL_PRODUCT_SOURCE_FILE: str = "app/rag/seed_products.json"

    # 현재 사용하는 임베딩 모델명
    FINANCIAL_PRODUCT_EMBEDDING_MODEL: str = "jhgan/ko-sroberta-multitask"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()