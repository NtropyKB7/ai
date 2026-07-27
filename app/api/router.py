from fastapi import APIRouter
from app.schemas.llm import LLMTestRequest, LLMTestResponse
from app.services.llm_service import llm_service
from app.rag.chroma_client import chroma_manager

router = APIRouter()

@router.get("/health")
def health_check():
    """
    서버 구동 상태 및 Chroma Vector DB 접속 핑을 확인하는 헬스체크 엔드포인트
    """
    chroma_status = chroma_manager.heartbeat()
    return {
        "status": "ok",
        "chroma_heartbeat": chroma_status
    }

@router.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """
    프롬프트를 수신하여 gpt-4o-mini 모델의 정상 연동 여부를 테스트하는 비동기 엔드포인트
    """
    result = await llm_service.generate_test(request.prompt)
    return LLMTestResponse(result=result)