import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
    force=True,
)

# FastAPI 앱 메인 인스턴스 생성 및 Swagger API 문서 타이틀 설정
app = FastAPI(
    title="FastAPI AI Server",
    version="1.0.0",
    docs_url="/docs", # Swagger UI 접근 경로
    redoc_url="/redoc"
)

# CORS 설정 (메인 Spring 서버와의 이종 간 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발 및 초기 세팅 단계 : 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"], # GET, POST, OPTIONS 등 모든 HTTP 메서드 허용
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router)
