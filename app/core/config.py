import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    CHROMA_DB_DIR: str = "./.chroma"

    class Config:
        env_file = ".env"

settings = Settings()