from fastapi import APIRouter, HTTPException, status

from app.rag.chroma_client import chroma_manager
from app.schemas.llm import LLMTestRequest, LLMTestResponse
from app.schemas.product import (
    ProductRecommendationRequest,
    ProductRecommendationResponse,
)
from app.schemas.transaction import (
    TransactionClassificationData,
    TransactionClassificationRequest,
    TransactionClassificationResponse,
)
from app.services.llm_service import llm_service
from app.services.product_knowledge_pipeline_service import (
    product_knowledge_pipeline_service,
)
from app.services.recommendation_service import recommendation_service
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)

router = APIRouter()

transaction_classification_service = TransactionClassificationService()


@router.get("/health")
def health_check():
    """
    서버 상태와 ChromaDB 연결 상태를 반환합니다.
    """
    chroma_status = chroma_manager.heartbeat()

    return {
        "status": "ok",
        "chroma_heartbeat": chroma_status,
    }


@router.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """
    LLM 연동 테스트 API입니다.
    """
    result = await llm_service.generate_test(request.prompt)
    return LLMTestResponse(result=result)


# =========================================================================
# 금융상품 추천 및 지식 DB 인덱싱 API
# =========================================================================

@router.post("/api/v1/products/seed", status_code=status.HTTP_201_CREATED)
def seed_products():
    """
    금융상품 지식 DB 갱신 파이프라인을 실행합니다.

    현재는 seed_products.json을 원천 데이터로 사용합니다.
    이후 외부 수집 경로가 추가되더라도 이 API는 동일한 파이프라인 진입점으로 유지할 수 있습니다.
    """
    try:
        result = product_knowledge_pipeline_service.refresh_from_seed_file()

        return {
            "message": "금융상품 지식 DB 갱신 완료",
            "source_count": result["source_count"],
            "normalized_count": result["normalized_count"],
            "upserted_count": result["upserted_count"],
        }

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"지식 DB 갱신 중 오류 발생: {str(error)}",
        )


@router.post(
    "/api/v1/recommend",
    response_model=ProductRecommendationResponse,
)
async def recommend_product(request: ProductRecommendationRequest):
    """
    월별 집계 데이터를 기반으로 금융상품 추천 결과를 생성합니다.
    """
    try:
        response = await recommendation_service.generate_recommendation(request)
        return response

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG 상품 추천 처리 중 오류 발생: {str(error)}",
        )


# =========================================================================
# 소비 내역 일괄 분류 API
# =========================================================================

@router.post(
    "/api/v1/classify-transactions",
    response_model=TransactionClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="소비 내역 일괄 분류",
    description=(
        "Spring AI-service가 전달한 거래 내역 목록을 규칙 기반으로 분류합니다. "
        "규칙으로 분류하기 어려운 거래만 LLM 보조 분류를 수행합니다. "
        "FastAPI는 분류 결과만 반환하며 Spring RDB에 직접 저장하지 않습니다."
    ),
)
async def classify_transactions(
    request: TransactionClassificationRequest,
) -> TransactionClassificationResponse:
    """
    거래 내역 여러 건을 일괄 분류합니다.
    """
    results = await transaction_classification_service.classify_transactions_with_llm(
        request.transactions,
    )

    return TransactionClassificationResponse(
        success=True,
        status_code=status.HTTP_200_OK,
        message="소비 내역 분류에 성공했습니다.",
        data=TransactionClassificationData(
            results=results,
        ),
    )