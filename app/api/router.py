from fastapi import APIRouter, HTTPException, status
from app.schemas.llm import LLMTestRequest, LLMTestResponse
from app.schemas.product import ProductRecommendationRequest, ProductRecommendationResponse
from app.services.llm_service import llm_service
from app.services.recommendation_service import recommendation_service
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

# =========================================================================
# 📌 [신규 추가] RAG 금융상품 추천 및 지식 DB 인덱싱 엔드포인트
# =========================================================================

@router.post("/api/v1/products/seed", status_code=status.HTTP_201_CREATED)
def seed_products():
    """
    [초기 세팅/관리자 전용 API]
    seed_products.json 내 금융상품 지식 데이터를 읽어 Chroma Vector DB에 임베딩 인덱싱을 수행합니다.
    """
    try:
        count = chroma_manager.seed_financial_products()
        return {"message": "금융상품 지식 DB 임베딩 적재 완료", "seeded_count": count}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"지식 DB 적재 중 오류 발생: {str(e)}"
        )

@router.post("/api/v1/recommend", response_model=ProductRecommendationResponse)
async def recommend_product(request: ProductRecommendationRequest):
    """
    [Spring 메인 서버 연동용 핵심 API]
    유저의 당월 재무 스냅샷 데이터를 전달받아 RAG 검색, 시뮬레이션 연산, 
    LLM 추론을 거쳐 맞춤형 1개 상품 추천 및 AI 코칭 리포트를 반환합니다.
    """
    try:
        response = await recommendation_service.generate_recommendation(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"RAG 상품 추천 처리 중 오류 발생: {str(e)}"
        )