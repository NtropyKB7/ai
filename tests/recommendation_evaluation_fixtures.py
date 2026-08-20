"""Synthetic, non-production evaluation assets for recommendation regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.schemas.product import ProductRecommendationRequest


class EvaluationStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    KNOWN_GAP = "KNOWN_GAP"


@dataclass(frozen=True)
class ExpectedDirection:
    allowed_product_types: frozenset[str] = frozenset({"CARD", "SAVINGS"})
    forbidden_product_ids: frozenset[str] = frozenset()
    fallback_required: bool = False
    note: str = ""


@dataclass(frozen=True)
class RecommendationScenario:
    scenario_id: str
    input_summary: str
    request: ProductRecommendationRequest
    candidates: tuple[dict[str, Any], ...]
    expected_direction: ExpectedDirection
    evaluation_status: EvaluationStatus = EvaluationStatus.SUPPORTED
    current_behavior: str = "deterministic recommendation"
    failure_reason: str | None = None
    llm_mode: str = "success"


@dataclass
class EvaluationResult:
    scenario_id: str
    input_summary: str
    search_candidates: list[str]
    filter_passed_candidates: list[str] | None
    reranked_result: list[str]
    final_recommended_product: str | None
    simulated_extra_income: int | None
    fallback_used: bool
    current_behavior: str
    expected_direction: ExpectedDirection
    evaluation_status: EvaluationStatus
    validation_passed: bool
    failure_reason: str | None = None


class FakeChromaManager:
    def __init__(self, products: tuple[dict[str, Any], ...] | list[dict[str, Any]]):
        self.products = [dict(product) for product in products]
        self.calls: list[dict[str, Any]] = []

    def search_products(self, query_text: str, n_results: int = 5):
        self.calls.append({"query_text": query_text, "n_results": n_results})
        return [dict(product) for product in self.products[:n_results]]


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, mode: str = "success"):
        self.mode = mode
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.mode == "error":
            raise RuntimeError("synthetic LLM failure")
        if self.mode == "invalid_json":
            return _FakeLLMResponse("this is not JSON")
        return _FakeLLMResponse(
            """{
              "reasoning": "합성 입력과 선택된 상품을 연결한 안전한 추천 사유입니다.",
              "financial_activity_insight": "합성 월간 소득과 소비 흐름을 요약한 인사이트입니다.",
              "financial_type": "균형 관리형",
              "job_insight": "합성 잡별 소득과 근무 정보에 기반한 인사이트입니다.",
              "future_income_trend": "현재 흐름을 유지하며 다음 달 자금 계획을 점검하세요."
            }"""
        )


def card(
    product_id: str,
    categories: tuple[str, ...],
    *,
    discount_rate: float = 0.1,
    max_benefit: int = 30_000,
    **extra_details,
) -> dict[str, Any]:
    details = {
        "discount_rate": discount_rate,
        "discount_categories": list(categories),
        "maxMonthlyBenefit": max_benefit,
        **extra_details,
    }
    return {
        "product_id": product_id,
        "product_name": f"합성 {product_id} 카드",
        "product_type": "CARD",
        "provider": "합성 카드사",
        "summary": f"{', '.join(categories)} 소비 할인",
        "target_group": "합성 평가 사용자",
        "njob_trend_tip": "합성 소비 절감 팁",
        "details": details,
    }


def savings(
    product_id: str,
    *,
    interest_rate: float = 4.5,
    max_monthly_amount: int = 1_000_000,
    **extra_details,
) -> dict[str, Any]:
    details = {
        "interest_rate": interest_rate,
        "maxMonthlyAmount": max_monthly_amount,
        **extra_details,
    }
    return {
        "product_id": product_id,
        "product_name": f"합성 {product_id} 적금",
        "product_type": "SAVINGS",
        "provider": "합성 은행",
        "summary": "합성 자유 적립식 적금",
        "target_group": "합성 평가 사용자",
        "njob_trend_tip": "합성 부수입 적립 팁",
        "details": details,
    }


FOOD_CARD = card("SYN_CARD_FOOD", ("FOOD",))
SHOPPING_CARD = card("SYN_CARD_SHOPPING", ("SHOPPING",))
GENERAL_CARD = card("SYN_CARD_GENERAL", ("OTHER",), discount_rate=0.03)
SAVINGS_STANDARD = savings("SYN_SAVINGS_STANDARD", interest_rate=4.5)


def request(
    *,
    income: int,
    expense: int,
    funds: int,
    category: str = "FOOD",
    category_amount: int | None = None,
    income_change_rate: float | None = None,
    volatility: float | None = None,
    jobs: list[dict[str, Any]] | None = None,
) -> ProductRecommendationRequest:
    amount = expense if category_amount is None else category_amount
    category_json = (
        f'[{{"category":"{category}","displayName":"합성 {category}",'
        f'"amount":{amount},"ratio":0.5}}]'
    )
    return ProductRecommendationRequest(
        user_id=900_001,
        year_month="2099-01",
        total_income=income,
        total_expense=expense,
        available_funds=funds,
        category_expenses=category_json,
        previous_month_income=None,
        income_change_amount=None,
        income_change_rate=income_change_rate,
        income_volatility=volatility,
        job_insight_inputs=jobs or [],
    )


SCENARIOS = (
    RecommendationScenario(
        "stable_surplus", "안정 소득과 충분한 가용자금",
        request(income=4_000_000, expense=2_000_000, funds=2_000_000),
        (GENERAL_CARD, SAVINGS_STANDARD),
        ExpectedDirection(frozenset({"SAVINGS"}), note="적금 추천 가능"),
    ),
    RecommendationScenario(
        "volatile_multi_job", "소득 변동성이 큰 합성 N잡 사용자",
        request(income=3_500_000, expense=2_000_000, funds=1_500_000, volatility=0.35,
                jobs=[{"jobId": 901, "jobName": "합성 주업", "incomeAmount": 2_000_000},
                      {"jobId": 902, "jobName": "합성 부업", "incomeAmount": 1_500_000}]),
        (GENERAL_CARD, SAVINGS_STANDARD),
        ExpectedDirection(frozenset({"SAVINGS"}), note="현재 점수식의 변동성 반영 기록"),
    ),
    RecommendationScenario(
        "spending_pressure", "소득 대비 소비가 큰 사용자",
        request(income=2_000_000, expense=1_900_000, funds=100_000),
        (SAVINGS_STANDARD, FOOD_CARD),
        ExpectedDirection(frozenset({"CARD"}), note="무리한 적금보다 카드 우선"),
    ),
    RecommendationScenario(
        "zero_available_funds", "가용자금이 0인 사용자",
        request(income=2_000_000, expense=2_000_000, funds=0),
        (SAVINGS_STANDARD, FOOD_CARD),
        ExpectedDirection(frozenset({"CARD"}), note="월 납입 부담 회피"),
    ),
    RecommendationScenario(
        "negative_cash_flow", "가용자금이 음수인 사용자",
        request(income=2_000_000, expense=2_400_000, funds=-400_000),
        (SAVINGS_STANDARD, FOOD_CARD),
        ExpectedDirection(frozenset({"CARD"}), note="현금흐름 위험형"),
    ),
    RecommendationScenario(
        "food_heavy", "식비 소비 비중이 높은 사용자",
        request(income=3_000_000, expense=2_700_000, funds=300_000, category="FOOD"),
        (SHOPPING_CARD, FOOD_CARD),
        ExpectedDirection(frozenset({"CARD"}), frozenset({"SYN_CARD_SHOPPING"}), note="식비 혜택 연결"),
    ),
    RecommendationScenario(
        "shopping_heavy", "쇼핑 소비 비중이 높은 사용자",
        request(income=3_000_000, expense=2_700_000, funds=300_000, category="SHOPPING"),
        (FOOD_CARD, SHOPPING_CARD),
        ExpectedDirection(frozenset({"CARD"}), frozenset({"SYN_CARD_FOOD"}), note="쇼핑 혜택 연결"),
    ),
    RecommendationScenario(
        "insufficient_savings_capacity", "적금 납입 여력이 부족한 사용자",
        request(income=2_500_000, expense=2_300_000, funds=200_000),
        (SAVINGS_STANDARD, FOOD_CARD),
        ExpectedDirection(frozenset({"CARD"}), note="현재 유형·점수 기반 카드 우선"),
    ),
    RecommendationScenario(
        "llm_failure", "LLM 호출 실패 사용자",
        request(income=4_000_000, expense=2_000_000, funds=2_000_000),
        (SAVINGS_STANDARD,),
        ExpectedDirection(frozenset({"SAVINGS"}), note="텍스트 fallback 정상 지원"),
        llm_mode="error",
    ),
    RecommendationScenario(
        "llm_invalid_json", "LLM JSON 파싱 실패 사용자",
        request(income=2_000_000, expense=1_900_000, funds=100_000),
        (FOOD_CARD,),
        ExpectedDirection(frozenset({"CARD"}), note="텍스트 fallback 정상 지원"),
        llm_mode="invalid_json",
    ),
    RecommendationScenario(
        "no_search_candidates", "검색 후보가 없는 사용자",
        request(income=3_000_000, expense=2_000_000, funds=1_000_000),
        (),
        ExpectedDirection(fallback_required=True, note="안전한 fallback 필요"),
        EvaluationStatus.KNOWN_GAP,
        "ValueError and HTTP 500",
        "현재 응답 DTO가 recommended_product를 필수로 요구함",
    ),
    RecommendationScenario(
        "inactive_product", "비활성 상품이 검색된 사용자",
        request(income=4_000_000, expense=2_000_000, funds=2_000_000),
        (savings("SYN_SAVINGS_INACTIVE", interest_rate=9.9, is_active=False), SAVINGS_STANDARD),
        ExpectedDirection(forbidden_product_ids=frozenset({"SYN_SAVINGS_INACTIVE"}), note="비활성 상품 제외 필요"),
        EvaluationStatus.KNOWN_GAP,
        "inactive metadata is not filtered",
        "현재 추천 파이프라인에 명시적인 상품 필터링 단계가 없음",
    ),
    RecommendationScenario(
        "eligibility_mismatch", "가입 조건을 충족하지 못한 사용자",
        request(income=2_500_000, expense=1_000_000, funds=1_500_000),
        (savings("SYN_SAVINGS_INELIGIBLE", interest_rate=9.0, minimum_income=8_000_000), SAVINGS_STANDARD),
        ExpectedDirection(forbidden_product_ids=frozenset({"SYN_SAVINGS_INELIGIBLE"}), note="가입 조건 불일치 제외 필요"),
        EvaluationStatus.KNOWN_GAP,
        "eligibility metadata is not filtered",
        "현재 추천 파이프라인에 명시적인 상품 필터링 단계가 없음",
    ),
    RecommendationScenario(
        "ended_product", "판매 종료 상품이 검색된 사용자",
        request(income=4_000_000, expense=2_000_000, funds=2_000_000),
        (savings("SYN_SAVINGS_ENDED", interest_rate=9.5, sale_status="ENDED"), SAVINGS_STANDARD),
        ExpectedDirection(forbidden_product_ids=frozenset({"SYN_SAVINGS_ENDED"}), note="판매 종료 상품 제외 필요"),
        EvaluationStatus.KNOWN_GAP,
        "sale status metadata is not filtered",
        "현재 추천 파이프라인에 명시적인 상품 필터링 단계가 없음",
    ),
)
