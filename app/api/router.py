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
from app.services.recommendation_service import recommendation_service
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


# FastAPI 라우터 객체입니다.
router = APIRouter()

# 규칙 기반 소비 내역 분류 서비스 객체를 한 번 생성합니다.
# 요청마다 새로 만들 필요 없이 재사용합니다.
transaction_classification_service = TransactionClassificationService()


@router.get("/health")
def health_check():
    """
    서버 구동 상태와 Chroma Vector DB 연결 상태를 확인합니다.
    """

    chroma_status = chroma_manager.heartbeat()

    return {
        "status": "ok",
        "chroma_heartbeat": chroma_status,
    }


@router.post("/api/v1/test-llm", response_model=LLMTestResponse)
async def test_llm(request: LLMTestRequest):
    """
    gpt-4o-mini 모델 연동 여부를 테스트하는 API입니다.
    """

    result = await llm_service.generate_test(request.prompt)

    return LLMTestResponse(result=result)


# =========================================================================
# 금융상품 추천 및 지식 DB 인덱싱 API
# =========================================================================

@router.post("/api/v1/products/seed", status_code=status.HTTP_201_CREATED)
def seed_products():
    """
    seed_products.json의 금융상품 데이터를
    Chroma Vector DB에 임베딩하여 저장합니다.
    """

    try:
        count = chroma_manager.seed_financial_products()

        return {
            "message": "금융상품 지식 DB 임베딩 적재 완료",
            "seeded_count": count,
        }

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"지식 DB 적재 중 오류 발생: {str(error)}",
        )


@router.post(
    "/api/v1/recommend",
    response_model=ProductRecommendationResponse,
)
async def recommend_product(request: ProductRecommendationRequest):
    """
    재무 스냅샷을 기반으로 RAG 검색과 LLM 추론을 수행하여
    금융상품 추천 및 코칭 리포트를 반환합니다.
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
        "FastAPI는 분류 결과만 반환하며 Spring RDB에 직접 저장하지 않습니다."
    ),
)
def classify_transactions(
    request: TransactionClassificationRequest,
) -> TransactionClassificationResponse:
    """
    거래 내역 여러 건을 한 번에 분류하는 API입니다.

    처리 흐름:
    1. Spring이 transactions 목록을 FastAPI로 전달합니다.
    2. 규칙 기반 분류 서비스를 호출합니다.
    3. 거래별 분류 결과를 공통 응답 구조로 반환합니다.
    4. Spring account-service가 transactionId 기준으로 TXN_ANALYSIS에 저장합니다.
    """

    # 요청에 포함된 모든 거래를 한 번에 분류합니다.
    results = transaction_classification_service.classify_transactions(
        request.transactions,
    )

    # 팀 공통 API 응답 형식으로 감싸서 반환합니다.
    return TransactionClassificationResponse(
        success=True,
        status_code=status.HTTP_200_OK,
        message="소비 내역 분류에 성공했습니다.",
        data=TransactionClassificationData(
            results=results,
        ),
    )