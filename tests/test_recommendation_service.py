import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.schemas.product import CategoryExpenseSummary, ProductRecommendationRequest
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