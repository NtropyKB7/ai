import json
import asyncio
from dataclasses import replace
from decimal import Decimal

import pytest

from app.schemas.product import CategoryExpenseSummary, ProductRecommendationRequest
from app.services.financial_product_interest_service import FinancialProductInterestService
from app.services.personalized_product_scoring_service import (
    EvidenceType,
    PersonalizedProductScoringService,
    PersonalizedScorePolicy,
    percentile_ranks,
)
from app.services.recommendation_profile_service import RecommendationProfileService
from tests.recommendation_evaluation_fixtures import FakeLLM


def request(**kwargs):
    base = dict(
        user_id=1, year_month="2026-08", total_income=4_000_000,
        total_expense=2_000_000, available_funds=2_000_000,
        category_expenses="[]",
    )
    base.update(kwargs)
    return ProductRecommendationRequest(**base)


def product(product_id, product_type, rate, term=12, reserve=None, **details):
    option = {
        "interest_rate_type": "S", "term_months": term,
        "base_interest_rate": rate, "preferred_interest_rate": rate,
    }
    if reserve is not None:
        option["reserve_type"] = reserve
    return {
        "product_id": product_id, "product_name": product_id,
        "product_type": product_type, "provider": "합성은행", "summary": "",
        "target_group": None, "njob_trend_tip": None,
        "details": {"options": [option], **details},
    }


def test_weights_and_percentiles_are_bounded_and_tie_safe():
    policy = PersonalizedScorePolicy()
    assert sum((policy.financial_benefit_weight, policy.semantic_relevance_weight,
                policy.financial_suitability_weight, policy.condition_match_weight)) == Decimal("1")
    assert percentile_ranks([1, 2, 2, 3]) == [0.0, 1 / 3, 1 / 3, 1.0]
    assert percentile_ranks([.4, .4, .9]) == [0.0, 0.0, 1.0]


def test_profile_budget_and_all_input_aggregation():
    req = request(
        available_funds=101, previous_month_income=3_000_000,
        income_change_amount=1_000_000, income_volatility=0.3,
        job_insight_inputs=[
            {"jobName": "직장인", "incomeAmount": 3_000_000, "incomeRatio": .75,
             "totalWorkMinutes": 1000, "workDays": 10, "averageFatigue": 3, "latestFatigue": 4},
            {"jobName": "프리랜서", "incomeAmount": 1_000_000, "incomeRatio": .25,
             "totalWorkMinutes": 500, "workDays": 5, "averageFatigue": 5, "latestFatigue": 5},
        ],
    )
    profile = RecommendationProfileService().build(req, [CategoryExpenseSummary(category="FOOD", amount=1)])
    assert profile.investment_budget == 50
    assert profile.multiple_income_sources
    assert profile.income_concentration == .75
    assert profile.total_work_minutes == 1500
    assert profile.total_work_days == 15
    assert profile.average_fatigue == 4
    assert profile.latest_fatigue == 5


def test_semantic_is_zero_without_personalization_signals():
    profile = RecommendationProfileService().build(
        request(total_income=0, total_expense=0, available_funds=0), []
    )
    profile = replace(profile, investment_budget=1)
    ranked = PersonalizedProductScoringService().score_all(
        [product("A", "DEPOSIT", 3), product("B", "DEPOSIT", 4)],
        profile, {"A": .9, "B": .1},
    )
    assert all(item.semantic_relevance_score == 0 for item in ranked)


def test_lower_rate_can_win_with_structural_and_condition_fit():
    req = request(income_volatility=.4, job_insight_inputs=[
        {"jobName": "프리랜서", "incomeAmount": 2_000_000},
        {"jobName": "직장인", "incomeAmount": 2_000_000},
    ])
    profile = RecommendationProfileService().build(req, [])
    candidates = [
        product("HIGH", "DEPOSIT", 5.0),
        product("FIT", "SAVINGS", 4.8, reserve="F",
                special_conditions_text="N잡 복수 소득 프리랜서 대상"),
        product("LOW1", "DEPOSIT", 2.0),
        product("LOW2", "DEPOSIT", 3.0),
        product("LOW3", "DEPOSIT", 4.0),
    ]
    ranked = PersonalizedProductScoringService().score_all(
        candidates, profile,
        {"HIGH": 0.1, "FIT": 0.9, "LOW1": 0.0, "LOW2": 0.0, "LOW3": 0.0},
    )
    assert ranked[0].product["product_id"] == "FIT"
    assert any(ev.evidence_type == EvidenceType.MULTI_INCOME_MATCH for ev in ranked[0].evidences)


def test_uncontrolled_job_and_unsupported_category_do_not_add_condition_points():
    req = request(job_insight_inputs=[{"jobName": "배달", "incomeAmount": 1_000_000}])
    categories = [CategoryExpenseSummary(category="UNKNOWN", displayName="여행", amount=1)]
    profile = RecommendationProfileService().build(req, categories)
    scored = PersonalizedProductScoringService().score_all(
        [product("A", "DEPOSIT", 3, special_conditions_text="배달 행사 여행")], profile, {"A": .5}
    )[0]
    assert scored.condition_match_score == 0


def test_category_needs_both_controlled_user_code_and_product_evidence():
    profile = RecommendationProfileService().build(
        request(), [CategoryExpenseSummary(category="LEISURE", amount=100, ratio=.8)]
    )
    no_evidence = product("A", "DEPOSIT", 3)
    evidence = product("B", "DEPOSIT", 3, special_conditions_text="여행 숙박 우대")
    scored = PersonalizedProductScoringService().score_all(
        [no_evidence, evidence], profile, {"A": 0, "B": 0}
    )
    values = {item.product["product_id"]: item.condition_match_score for item in scored}
    assert values["A"] == 0
    assert values["B"] > 0


def test_duplicate_expressions_create_one_evidence_and_scores_are_bounded():
    profile = RecommendationProfileService().build(
        request(job_insight_inputs=[
            {"jobName": "직장인", "incomeAmount": 2_000_000},
            {"jobName": "프리랜서", "incomeAmount": 2_000_000},
        ]), []
    )
    item = product("A", "SAVINGS", 4, reserve="F",
                   special_conditions_text="N잡 N잡 복수소득 복수 소득 프리랜서")
    scored = PersonalizedProductScoringService().score_all([item], profile, {"A": .5})[0]
    keys = [(ev.evidence_type, ev.normalized_key) for ev in scored.evidences]
    assert len(keys) == len(set(keys))
    assert all(0 <= value <= 1 for value in (
        scored.financial_benefit_score, scored.semantic_relevance_score,
        scored.financial_suitability_score, scored.condition_match_score,
        scored.personalized_score,
    ))


def test_all_96_candidates_are_rescored_and_input_order_does_not_change_winner():
    profile = RecommendationProfileService().build(request(income_volatility=.3), [])
    candidates = [product(f"S{i:02}", "SAVINGS", 2 + i / 100, reserve="F") for i in range(58)]
    candidates += [product(f"D{i:02}", "DEPOSIT", 2 + i / 100) for i in range(38)]
    similarities = {item["product_id"]: index / 100 for index, item in enumerate(candidates)}
    service = PersonalizedProductScoringService()
    first = service.score_all(candidates, profile, similarities)
    second = service.score_all(list(reversed(candidates)), profile, similarities)
    assert len(first) == 96
    assert first[0].product["product_id"] == second[0].product["product_id"]


def test_interest_formulas_floor_only_final_result():
    service = FinancialProductInterestService()
    assert service.monthly_interest("SAVINGS", 500_000, Decimal("4.2"), 12) == 11_375
    assert service.monthly_interest("DEPOSIT", 500_000, Decimal("4.2"), 12) == 1_750


def test_recommendation_returns_one_deposit_and_calculates_only_selected(
    isolated_recommendation_runtime,
):
    candidates = [
        product("S", "SAVINGS", 3.0, reserve="S"),
        product("D", "DEPOSIT", 4.0),
    ]
    class Chroma:
        def search_finlife_products_with_scores(self, query):
            return list(reversed(candidates)), {"S": .1, "D": .8}
    service = isolated_recommendation_runtime.RecommendationService(
        llm=FakeLLM(), chroma=Chroma()
    )
    calls = []
    original = service.interest_service.monthly_interest
    def spy(**kwargs):
        calls.append(kwargs["product_type"])
        return original(**kwargs)
    service.interest_service.monthly_interest = spy
    response = asyncio.run(service.generate_recommendation(request()))
    assert response.recommended_product.product_type == "DEPOSIT"
    assert calls == ["DEPOSIT"]
    assert response.simulated_extra_income >= 0
    assert "전체 보유 목돈" in response.reasoning
