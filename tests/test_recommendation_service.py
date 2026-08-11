import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.schemas.product import (
    CategoryExpenseSummary,
    FinancialProductSchema,
    ProductRecommendationRequest,
)
from app.services.recommendation_service import RecommendationService


def test_parse_category_expenses_list_json():
    service = RecommendationService()

    raw_json = json.dumps(
        [
            {
                "category": "FOOD",
                "displayName": "식비",
                "amount": 800000,
                "ratio": 0.31,
            }
        ],
        ensure_ascii=False,
    )

    result = service._parse_category_expenses(raw_json)

    assert len(result) == 1
    assert result[0].category == "FOOD"
    assert result[0].displayName == "식비"
    assert result[0].amount == 800000
    assert result[0].ratio == 0.31


def test_parse_category_expenses_invalid_json():
    service = RecommendationService()

    result = service._parse_category_expenses("not-json")

    assert result == []


def test_classify_financial_type_savings_capacity():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=4_000_000,
        total_expense=2_500_000,
        available_funds=1_500_000,
    )

    assert service._classify_financial_type(request) == "저축 여력형"


def test_classify_financial_type_negative_cash_flow():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=2_000_000,
        total_expense=2_500_000,
        available_funds=-500_000,
    )

    assert service._classify_financial_type(request) == "현금흐름 위험형"


def test_build_job_insight_with_job_insight_inputs():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=4_000_000,
        total_expense=2_500_000,
        available_funds=1_500_000,
        job_insight_inputs=[
            {
                "jobId": 1,
                "jobName": "배달",
                "incomeAmount": 1_000_000,
                "incomeRatio": 0.25,
                "totalWorkMinutes": 2400,
                "workDays": 10,
                "averageFatigue": 3.2,
                "latestFatigue": 4,
            }
        ],
    )

    result = service._build_job_insight(request.job_insight_inputs)

    assert "배달" in result
    assert "1,000,000원" in result
    assert "40.0시간" in result
    assert "3.2" in result


def test_build_product_search_query():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=4_000_000,
        total_expense=2_500_000,
        available_funds=1_500_000,
        income_change_rate=0.1,
        income_volatility=0.18,
    )

    categories = [
        CategoryExpenseSummary(
            category="FOOD",
            displayName="식비",
            amount=800000,
            ratio=0.32,
        )
    ]

    result = service._build_product_search_query(
        request=request,
        category_expenses=categories,
        financial_type="저축 여력형",
    )

    assert "저축 여력형" in result
    assert "1500000원" in result
    assert "FOOD" in result
    assert "0.18" in result


def test_parse_llm_json_with_markdown_block():
    service = RecommendationService()

    result = service._parse_llm_json(
        """
```json
{
  "reasoning": "추천 사유",
  "financial_activity_insight": "재무 인사이트",
  "financial_type": "저축 여력형",
  "job_insight": "잡 인사이트",
  "future_income_trend": "미래 소득 전망"
}
```
"""
    )

    assert result["reasoning"] == "추천 사유"
    assert result["financial_activity_insight"] == "재무 인사이트"
    assert result["financial_type"] == "저축 여력형"
    assert result["job_insight"] == "잡 인사이트"
    assert result["future_income_trend"] == "미래 소득 전망"


def test_parse_llm_json_invalid_text():
    service = RecommendationService()

    result = service._parse_llm_json("not-json")

    assert result == {}


def test_select_best_product_prefers_card_for_spending_pressure():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=2_000_000,
        total_expense=1_900_000,
        available_funds=100_000,
    )

    categories = [
        CategoryExpenseSummary(
            category="FOOD",
            displayName="식비",
            amount=900000,
            ratio=0.47,
        )
    ]

    candidates = [
        {
            "product_id": "SAVINGS_001",
            "product_name": "적금 상품",
            "product_type": "SAVINGS",
            "provider": "은행",
            "summary": "자유적금 상품",
            "target_group": "N잡러",
            "njob_trend_tip": "부수입 적립",
            "details": {"interest_rate": 4.0},
        },
        {
            "product_id": "CARD_001",
            "product_name": "생활 할인 카드",
            "product_type": "CARD",
            "provider": "카드사",
            "summary": "식비 할인 카드",
            "target_group": "생활비 관리형",
            "njob_trend_tip": "식비 절감",
            "details": {
                "discount_rate": 0.1,
                "discount_categories": ["FOOD"],
            },
        },
    ]

    result = service._select_best_product(
        products=candidates,
        request=request,
        category_expenses=categories,
        financial_type="소비 압박형",
    )

    assert result["product_id"] == "CARD_001"


def test_select_best_product_prefers_savings_for_surplus():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=4_000_000,
        total_expense=2_000_000,
        available_funds=2_000_000,
        income_change_rate=0.12,
    )

    categories = [
        CategoryExpenseSummary(
            category="FOOD",
            displayName="식비",
            amount=500000,
            ratio=0.25,
        )
    ]

    candidates = [
        {
            "product_id": "CARD_001",
            "product_name": "생활 할인 카드",
            "product_type": "CARD",
            "provider": "카드사",
            "summary": "식비 할인 카드",
            "target_group": "생활비 관리형",
            "njob_trend_tip": "식비 절감",
            "details": {
                "discount_rate": 0.1,
                "discount_categories": ["FOOD"],
            },
        },
        {
            "product_id": "SAVINGS_001",
            "product_name": "자유적금 상품",
            "product_type": "SAVINGS",
            "provider": "은행",
            "summary": "우대금리 적금 상품",
            "target_group": "N잡러",
            "njob_trend_tip": "부수입 적립",
            "details": {"interest_rate": 4.2},
        },
    ]

    result = service._select_best_product(
        products=candidates,
        request=request,
        category_expenses=categories,
        financial_type="저축 여력형",
    )

    assert result["product_id"] == "SAVINGS_001"


def test_calculate_simulated_extra_income_applies_card_cap():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=3_000_000,
        total_expense=2_000_000,
        available_funds=1_000_000,
    )

    categories = [
        CategoryExpenseSummary(
            category="FOOD",
            displayName="식비",
            amount=500000,
            ratio=0.25,
        )
    ]

    product = {
        "product_id": "CARD_001",
        "product_name": "생활 할인 카드",
        "product_type": "CARD",
        "provider": "카드사",
        "summary": "식비 할인 카드",
        "target_group": "생활비 관리형",
        "njob_trend_tip": "식비 절감",
        "details": {
            "discount_rate": 0.1,
            "discount_categories": ["FOOD"],
            "maxMonthlyBenefit": 30000,
        },
    }

    result = service._calculate_simulated_extra_income(
        product=product,
        request=request,
        category_expenses=categories,
    )

    assert result == 30000


def test_build_default_response_texts_contains_all_required_fields():
    service = RecommendationService()

    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-07",
        total_income=4_000_000,
        total_expense=2_500_000,
        available_funds=1_500_000,
    )

    product = FinancialProductSchema(
        product_id="SAVINGS_001",
        product_name="자유적금 상품",
        product_type="SAVINGS",
        provider="은행",
        summary="우대금리 적금 상품",
        target_group="N잡러",
        njob_trend_tip="부수입 적립",
        details={"interest_rate": 4.2},
    )

    texts = service._build_default_response_texts(
        request=request,
        product=product,
        simulated_extra_income=4200,
        financial_type="저축 여력형",
        financial_activity_insight="재무활동 요약",
        job_insight="잡 인사이트",
    )

    assert set(texts.keys()) == {
        "reasoning",
        "financial_activity_insight",
        "financial_type",
        "job_insight",
        "future_income_trend",
    }


def test_merge_llm_texts_with_defaults_keeps_default_when_llm_value_too_short():
    service = RecommendationService()

    default_texts = {
        "reasoning": "기본 추천 사유 문구입니다.",
        "financial_activity_insight": "기본 재무활동 문구입니다.",
        "financial_type": "저축 여력형",
        "job_insight": "기본 잡 인사이트 문구입니다.",
        "future_income_trend": "기본 미래 소득 문구입니다.",
    }

    llm_result = {
        "reasoning": "짧음",
        "financial_type": "저축 여력형.",
    }

    merged = service._merge_llm_texts_with_defaults(
        llm_result=llm_result,
        default_texts=default_texts,
    )

    assert merged["reasoning"] == "기본 추천 사유 문구입니다."
    assert merged["financial_type"] == "저축 여력형"
