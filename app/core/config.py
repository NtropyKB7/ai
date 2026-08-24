from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    환경 변수 및 앱 전역 설정 관리 클래스입니다.
    """

    OPENAI_API_KEY: str
    OPENAI_CLASSIFICATION_MODEL: str = "gpt-5-nano"
    OPENAI_GENERATION_MODEL: str = "gpt-4o-mini"
    OPENAI_CLASSIFICATION_MAX_CONCURRENCY: int = Field(
        default=4,
        ge=1,
        le=16,
    )
    OPENAI_CLASSIFICATION_HTTP_TIMEOUT_SECONDS: float = Field(
        default=25.0,
        gt=0,
        le=120.0,
    )
    OPENAI_CLASSIFICATION_TIMEOUT_SECONDS: float = Field(
        default=30.0,
        gt=0,
        le=120.0,
    )
    OPENAI_CLASSIFICATION_MAX_RETRIES: int = Field(
        default=0,
        ge=0,
        le=1,
    )
    OPENAI_CLASSIFICATION_CHUNK_SIZE: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    # Finlife HTTP access is optional until the explicit collector/CLI is used.
    FINLIFE_API_KEY: Optional[SecretStr] = None
    FINLIFE_BASE_URL: str = "https://finlife.fss.or.kr"
    FINLIFE_SAVINGS_PATH: str = "/finlifeapi/savingProductsSearch.json"
    FINLIFE_DEPOSIT_PATH: str = "/finlifeapi/depositProductsSearch.json"
    FINLIFE_CONNECT_TIMEOUT_SECONDS: float = 5.0
    FINLIFE_READ_TIMEOUT_SECONDS: float = 15.0
    FINLIFE_WRITE_TIMEOUT_SECONDS: float = 5.0
    FINLIFE_POOL_TIMEOUT_SECONDS: float = 5.0

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
