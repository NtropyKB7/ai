"""Issue #32: current-behavior evaluation and recommendation regressions.

KNOWN_GAP cases intentionally document unsupported production behavior.  They
must not be interpreted as successful fallback or correct recommendations.
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.recommendation_evaluation_fixtures import (
    EvaluationResult,
    EvaluationStatus,
    FakeChromaManager,
    FakeLLM,
    SCENARIOS,
)

pytestmark = pytest.mark.usefixtures("isolated_recommendation_runtime")


def run(coroutine):
    return asyncio.run(coroutine)


def evaluate_supported_scenario(scenario, runtime):
    fake_chroma = FakeChromaManager(scenario.candidates)
    fake_llm = FakeLLM(scenario.llm_mode)
    service = runtime.RecommendationService(llm=fake_llm, chroma=fake_chroma)
    response = run(service.generate_recommendation(scenario.request))
    selected_id = response.recommended_product.product_id

    validation_passed = (
        response.recommended_product.product_type
        in scenario.expected_direction.allowed_product_types
        and selected_id not in scenario.expected_direction.forbidden_product_ids
    )
    fallback_used = scenario.llm_mode in {"error", "invalid_json"}
    result = EvaluationResult(
        scenario_id=scenario.scenario_id,
        input_summary=scenario.input_summary,
        search_candidates=[item["product_id"] for item in scenario.candidates],
        # No production filtering stage exists; None makes that absence explicit.
        filter_passed_candidates=None,
        reranked_result=[selected_id],
        final_recommended_product=selected_id,
        simulated_extra_income=response.simulated_extra_income,
        fallback_used=fallback_used,
        current_behavior=scenario.current_behavior,
        expected_direction=scenario.expected_direction,
        evaluation_status=scenario.evaluation_status,
        validation_passed=validation_passed,
    )
    return response, fake_chroma, fake_llm, result


def test_evaluation_assets_define_at_least_twelve_synthetic_scenarios():
    assert len(SCENARIOS) >= 12
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == len(SCENARIOS)
    assert all(scenario.request.user_id == 900_001 for scenario in SCENARIOS)
    assert all(scenario.request.year_month == "2099-01" for scenario in SCENARIOS)
    assert {scenario.evaluation_status for scenario in SCENARIOS} == {
        EvaluationStatus.SUPPORTED,
        EvaluationStatus.KNOWN_GAP,
    }


@pytest.mark.parametrize(
    "scenario",
    [item for item in SCENARIOS if item.evaluation_status == EvaluationStatus.SUPPORTED],
    ids=lambda item: item.scenario_id,
)
def test_supported_recommendation_directions_are_deterministic(
    scenario, isolated_recommendation_runtime
):
    runtime = isolated_recommendation_runtime
    first_response, first_chroma, _, first_result = evaluate_supported_scenario(
        scenario, runtime
    )
    second_response, _, _, second_result = evaluate_supported_scenario(
        scenario, runtime
    )

    assert first_result.validation_passed is True
    assert first_result.filter_passed_candidates is None
    assert first_response.recommended_product == second_response.recommended_product
    assert first_response.simulated_extra_income == second_response.simulated_extra_income
    assert first_result.final_recommended_product == second_result.final_recommended_product
    assert first_chroma.calls[0]["n_results"] == 5


@pytest.mark.parametrize("llm_mode", ["error", "invalid_json"])
def test_llm_failure_modes_use_safe_text_fallback_without_changing_product(
    llm_mode, isolated_recommendation_runtime
):
    scenario = next(item for item in SCENARIOS if item.llm_mode == llm_mode)
    response, _, fake_llm, result = evaluate_supported_scenario(
        scenario, isolated_recommendation_runtime
    )

    assert result.fallback_used is True
    assert response.recommended_product.product_id == scenario.candidates[0]["product_id"]
    assert response.reasoning
    assert response.financial_activity_insight
    assert response.job_insight
    assert response.future_income_trend
    assert all(
        candidate["product_name"] not in response.reasoning
        for candidate in scenario.candidates[1:]
    )
    assert len(fake_llm.prompts) == 1


def test_known_gap_no_candidates_currently_raises_and_requires_safe_fallback(
    isolated_recommendation_runtime,
):
    """KNOWN_GAP: candidate absence is an exception, not a valid fallback."""
    scenario = next(item for item in SCENARIOS if item.scenario_id == "no_search_candidates")
    service = isolated_recommendation_runtime.RecommendationService(
        llm=FakeLLM(),
        chroma=FakeChromaManager(scenario.candidates),
    )

    with pytest.raises(ValueError, match="추천 후보 상품이 비어 있습니다"):
        run(service.generate_recommendation(scenario.request))

    assert scenario.evaluation_status == EvaluationStatus.KNOWN_GAP
    assert scenario.current_behavior == "ValueError and HTTP 500"
    assert scenario.expected_direction.fallback_required is True
    assert scenario.failure_reason == "현재 응답 DTO가 recommended_product를 필수로 요구함"


@pytest.mark.parametrize(
    "scenario",
    [item for item in SCENARIOS if item.scenario_id in {
        "inactive_product", "eligibility_mismatch", "ended_product"
    }],
    ids=lambda item: item.scenario_id,
)
def test_known_gap_filter_conditions_are_observed_not_presented_as_supported(
    scenario, isolated_recommendation_runtime
):
    """KNOWN_GAP: unsupported filtering is recorded without asserting it works."""
    fake_chroma = FakeChromaManager(scenario.candidates)
    service = isolated_recommendation_runtime.RecommendationService(
        llm=FakeLLM(), chroma=fake_chroma
    )
    response = run(service.generate_recommendation(scenario.request))
    observed_product_id = response.recommended_product.product_id

    result = EvaluationResult(
        scenario_id=scenario.scenario_id,
        input_summary=scenario.input_summary,
        search_candidates=[item["product_id"] for item in scenario.candidates],
        filter_passed_candidates=None,
        reranked_result=[observed_product_id],
        final_recommended_product=observed_product_id,
        simulated_extra_income=response.simulated_extra_income,
        fallback_used=False,
        current_behavior=scenario.current_behavior,
        expected_direction=scenario.expected_direction,
        evaluation_status=EvaluationStatus.KNOWN_GAP,
        validation_passed=(observed_product_id not in scenario.expected_direction.forbidden_product_ids),
        failure_reason=scenario.failure_reason,
    )

    assert result.evaluation_status == EvaluationStatus.KNOWN_GAP
    assert result.filter_passed_candidates is None
    assert result.validation_passed is False
    assert result.failure_reason == "현재 추천 파이프라인에 명시적인 상품 필터링 단계가 없음"


def test_recommendation_request_and_response_dto_contract_regression(
    isolated_recommendation_runtime,
):
    runtime = isolated_recommendation_runtime
    request_payload = {
        "user_id": 900001,
        "year_month": "2099-01",
        "total_income": 4_000_000,
        "total_expense": 2_000_000,
        "available_funds": 2_000_000,
        "category_expenses": "[]",
        "job_insight_inputs": [],
    }
    request_model = runtime.ProductRecommendationRequest.model_validate(request_payload)
    assert request_model.model_dump()["user_id"] == 900001

    response_model = runtime.ProductRecommendationResponse(
        recommended_product=runtime.FinancialProductSchema(
            product_id="SYN_SAVINGS_CONTRACT",
            product_name="합성 계약 적금",
            product_type="SAVINGS",
            provider="합성 은행",
            summary="합성 계약 검증",
        ),
        simulated_extra_income=1000,
        reasoning="합성 추천 사유",
        financial_activity_insight="합성 재무 인사이트",
        financial_type="저축 여력형",
        job_insight="합성 잡 인사이트",
        future_income_trend="합성 미래 소득 인사이트",
    )
    dumped = response_model.model_dump()
    assert set(dumped) == {
        "recommended_product", "simulated_extra_income", "reasoning",
        "financial_activity_insight", "financial_type", "job_insight",
        "future_income_trend",
    }
    assert dumped["recommended_product"]["product_id"] == "SYN_SAVINGS_CONTRACT"


def test_recommendation_api_contract_with_fake_service(
    monkeypatch, isolated_recommendation_runtime
):
    runtime = isolated_recommendation_runtime
    scenario = next(item for item in SCENARIOS if item.scenario_id == "stable_surplus")
    service = runtime.RecommendationService(
        llm=FakeLLM(), chroma=FakeChromaManager(scenario.candidates)
    )
    monkeypatch.setattr(runtime.router_module, "recommendation_service", service)

    with TestClient(runtime.app) as client:
        response = client.post("/api/v1/recommend", json=scenario.request.model_dump())

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_product"]["product_id"] == "SYN_SAVINGS_STANDARD"
    assert isinstance(payload["simulated_extra_income"], int)
    assert all(payload[field] for field in (
        "reasoning", "financial_activity_insight", "financial_type",
        "job_insight", "future_income_trend",
    ))


def test_known_gap_no_candidates_api_currently_returns_http_500(
    monkeypatch, isolated_recommendation_runtime
):
    """KNOWN_GAP: HTTP 500 records current behavior; it is not a valid fallback."""
    scenario = next(item for item in SCENARIOS if item.scenario_id == "no_search_candidates")
    runtime = isolated_recommendation_runtime
    service = runtime.RecommendationService(llm=FakeLLM(), chroma=FakeChromaManager([]))
    monkeypatch.setattr(runtime.router_module, "recommendation_service", service)

    with TestClient(runtime.app) as client:
        response = client.post("/api/v1/recommend", json=scenario.request.model_dump())

    assert response.status_code == 500
    assert "추천 후보 상품이 비어 있습니다" in response.json()["detail"]
    assert scenario.evaluation_status == EvaluationStatus.KNOWN_GAP


def test_evaluation_uses_only_synthetic_products_and_no_secret_fields():
    serialized = json.dumps(
        [
            {
                "request": scenario.request.model_dump(),
                "candidates": scenario.candidates,
            }
            for scenario in SCENARIOS
        ],
        ensure_ascii=False,
    )
    assert "OPENAI_API_KEY" not in serialized
    assert ".env" not in serialized
    assert all(
        product["product_id"].startswith("SYN_")
        for scenario in SCENARIOS
        for product in scenario.candidates
    )


def test_collection_is_isolated_from_env_chroma_model_download_and_external_network(
    isolated_recommendation_runtime,
):
    runtime = isolated_recommendation_runtime
    assert Path.cwd() == runtime.temp_directory
    assert not (Path.cwd() / ".env").exists()
    assert type(runtime.chroma_manager.client).__name__ == "_IsolatedChromaClient"
    assert type(runtime.chroma_manager.embeddings).__name__ == "_IsolatedEmbeddingAdapter"

    with socket.socket() as network_socket:
        with pytest.raises(AssertionError, match="must not access external address"):
            network_socket.connect(("203.0.113.1", 443))
