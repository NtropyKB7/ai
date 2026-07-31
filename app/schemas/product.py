from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FinancialProductSchema(BaseModel):
    """
    FINANCIAL_PRODUCT 지식 DB 원본 및 추천 응답 매핑용 Pydantic DTO.
    공통 기본 메타데이터와 가변 상세 데이터를 분리하여 유연성을 확보합니다.
    """
    product_id: str = Field(..., description="상품 고유 식별 ID (예: 'CARD_001', 'SAVINGS_001')")
    product_name: str = Field(..., description="금융 상품명 (예: 'KB K-패스 라이더 혜택 카드')")
    product_type: str = Field(..., description="상품 유형 구분 ('CARD', 'SAVINGS')")
    provider: str = Field(..., description="상품 제공 금융사 (예: 'KB국민카드', 'KB국민은행')")
    summary: str = Field(..., description="RAG 벡터 임베딩 및 유사도 검색용 핵심 혜택 요약 문장")

    target_group: Optional[str] = Field(None, description="주 추천 대상 고객군 (예: '배달 라이더', 'N잡러')")
    njob_trend_tip: Optional[str] = Field(None, description="해당 상품과 연계 가능한 N잡 수익 창출/활용 팁 문구")

    # 카드(연회비, 실적 조건 등) 및 적금(금리, 기간 등)의 특화 속성을 JSON/Dict 형태로 유연하게 포장
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="상품 유형별 상세 가변 속성 (JSON 데이터)"
    )


class ProductRecommendationRequest(BaseModel):
    """
    Spring 메인 서버로부터 전달받는 유저 재무 스냅샷 요청 DTO.
    RAG 유사도 검색 및 시뮬레이션 연산의 입력값으로 활용됩니다.
    """
    user_id: int = Field(..., description="유저 PK")
    year_month: str = Field(..., description="조회 대상 연월 (예: '2026-06')")
    
    # 📌 (원 단위) 명시를 통해 수치 단위 혼선 방지
    available_funds: int = Field(..., description="당월 가용 자금 (총소득 - 총소비, 원 단위, 예: 500000)")
    
    # 📌 Optional 필드의 default 값을 default=0 으로 명확히 지정
    monthly_fuel_expense: Optional[int] = Field(
        default=0, 
        description="당월 주유/이동 관련 지출액 (원 단위, 예: 350000. 미입력 시 0)"
    )
    
    consumption_summary: str = Field(..., description="유저의 당월 소비 특성 텍스트 요약 (Vector DB 쿼리문으로 사용)")


class ProductRecommendationResponse(BaseModel):
    """
    FastAPI -> Spring 메인 서버로 최종 반환되는 RAG AI 금융상품 추천 응답 DTO.
    기획 요구사항(단일 추천 + 시뮬레이션 금액)이 반영되어 있습니다.
    """
    # RAG 검색을 통해 엄선된 단 1개의 최적 맞춤 상품
    recommended_product: FinancialProductSchema = Field(..., description="최적의 1개 맞춤 추천 상품")
    
    # 기획자 요구 기능: "이때 이 상품을 썼으면 벌었을/절약했을 예상 금액"
    simulated_extra_income: int = Field(
        ..., 
        description="유저가 이 상품을 지불/적립에 활용했을 때 예상되는 추가 이자 수익 또는 절감 금액 (원 단위)"
    )
    
    # LLM이 지식을 주입(Augment)받아 생성한 추천 사유 및 N잡 코칭
    reasoning: str = Field(..., description="상품 사용 시 이득을 직관적으로 설명하는 추천 사유 문구")
    future_income_trend: str = Field(..., description="유저 소비/소득 특성을 분석한 미래 수익 트렌드 및 N잡 코칭 문구")