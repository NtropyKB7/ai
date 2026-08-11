from typing import Any, Optional

from pydantic import BaseModel, Field


class FinancialProductSchema(BaseModel):
    """
    금융상품 지식 DB 원본과 추천 응답에 사용하는 DTO입니다.
    ChromaDB에서 검색된 금융상품 정보를 Spring AI-service로 반환할 때 사용합니다.
    """

    product_id: str = Field(..., description="상품 고유 식별 ID")
    product_name: str = Field(..., description="금융 상품명")
    product_type: str = Field(..., description="상품 유형. 예: CARD, SAVINGS")
    provider: str = Field(..., description="상품 제공 금융사")
    summary: str = Field(..., description="상품 핵심 혜택 요약")

    target_group: Optional[str] = Field(default=None, description="주 추천 대상 고객군")
    njob_trend_tip: Optional[str] = Field(default=None, description="N잡러 활용 팁")

    details: dict[str, Any] = Field(
        default_factory=dict,
        description="상품 유형별 상세 정보",
    )


class CategoryExpenseSummary(BaseModel):
    """
    Java AI-service가 category_expenses 문자열 안에 넣어 보내는
    카테고리별 소비 요약 DTO입니다.
    """

    category: str = Field(..., description="소비 카테고리 코드")
    displayName: Optional[str] = Field(default=None, description="화면 표시명")
    amount: int = Field(default=0, description="카테고리별 소비 금액")
    ratio: Optional[float] = Field(
        default=None,
        description="전체 소비 대비 비율. 0~1 사이 소수",
    )


class JobInsightInput(BaseModel):
    """
    Java AI-service가 jobId 기준으로 조합해서 전달하는
    잡별 AI 인사이트 생성용 입력 DTO입니다.

    FastAPI는 work-service 원본 DTO 전체를 받지 않고,
    AI 판단에 필요한 요약값만 받습니다.
    """

    jobId: Optional[int] = Field(default=None, description="잡 ID")
    jobName: Optional[str] = Field(default=None, description="잡 이름")
    incomeAmount: Optional[int] = Field(default=0, description="잡별 소득 금액")
    incomeRatio: Optional[float] = Field(
        default=None,
        description="전체 소득 대비 잡별 소득 비율. 0~1 사이 소수",
    )
    totalWorkMinutes: Optional[int] = Field(default=0, description="잡별 총 근무 시간")
    workDays: Optional[int] = Field(default=0, description="잡별 근무일수")
    averageFatigue: Optional[float] = Field(default=None, description="잡별 평균 피로도")
    latestFatigue: Optional[int] = Field(default=None, description="잡별 최근 피로도")


class ProductRecommendationRequest(BaseModel):
    """
    Spring AI-service가 FastAPI 추천 API로 전달하는 월별 집계 데이터입니다.

    원천 거래 전체가 아니라, 추천과 리포트 문구 생성에 필요한
    월별 집계 데이터만 받습니다.
    """

    user_id: int = Field(..., description="사용자 ID")
    year_month: str = Field(..., description="추천 대상 연월. 예: 2026-07")

    total_income: int = Field(default=0, description="해당 월 총소득")
    total_expense: int = Field(default=0, description="해당 월 총소비")
    available_funds: int = Field(
        default=0,
        description="가용자금. total_income - total_expense",
    )

    category_expenses: str = Field(
        default="[]",
        description="카테고리별 소비 요약 JSON 문자열",
    )

    previous_month_income: Optional[int] = Field(default=None, description="전월 총소득")
    income_change_amount: Optional[int] = Field(default=None, description="전월 대비 소득 증감액")
    income_change_rate: Optional[float] = Field(
        default=None,
        description="전월 대비 소득 증감률. 0~1 사이 소수",
    )
    income_volatility: Optional[float] = Field(default=None, description="최근 소득 변동성")

    job_insight_inputs: list[JobInsightInput] = Field(
        default_factory=list,
        description="잡별 소득·근무시간·피로도 기반 AI 인사이트 입력 목록",
    )


class ProductRecommendationResponse(BaseModel):
    """
    FastAPI가 Spring AI-service로 반환하는 금융상품 추천 결과 DTO입니다.
    """

    recommended_product: FinancialProductSchema = Field(..., description="추천 금융상품")
    simulated_extra_income: int = Field(..., description="예상 추가 수익 또는 절감 금액")
    reasoning: str = Field(..., description="추천 상품 사용 시 이득을 설명하는 문구")

    financial_activity_insight: str = Field(..., description="소비·소득 활동 요약 인사이트")
    financial_type: str = Field(..., description="사용자의 재무 유형")
    job_insight: str = Field(..., description="잡별 소득·근무시간·피로도 기반 인사이트")
    future_income_trend: str = Field(..., description="미래 소득 트렌드 및 N잡 코칭 문구")