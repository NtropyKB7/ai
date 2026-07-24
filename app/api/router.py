from fastapi import APIRouter
from app.schemas.llm import LLMTestRequest, LLMTestResponse
from app.services.llm_service import llm_service
from app.rag.chroma_client import chroma_manager

router = APIRouter()

@router.get("/health")
def health_check():
    chroma_status = chroma_manager.heartbeat()
    return {
        "status": "ok",
        "chroma_heartbeat": chroma_status
    }

@router.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    result = await llm_service.generate_test(request.prompt)
    return LLMTestResponse(result=result)