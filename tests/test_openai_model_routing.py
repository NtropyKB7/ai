import asyncio
import logging

from app.core.config import Settings
from app.api import router as router_module
from app.schemas.product import FinancialProductSchema, ProductRecommendationRequest
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    LLMTransactionClassificationResponse,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services import llm_service as llm_service_module
from app.services import recommendation_service as recommendation_service_module


class _FakeStructuredLLM:
    def __or__(self, other):
        return other


class _FakeChatOpenAI:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def with_structured_output(self, schema):
        return _FakeStructuredLLM()


def test_model_settings_have_task_specific_defaults():
    configured = Settings(OPENAI_API_KEY="synthetic-key", _env_file=None)

    assert configured.OPENAI_CLASSIFICATION_MODEL == "gpt-5-nano"
    assert configured.OPENAI_GENERATION_MODEL == "gpt-4o-mini"


def test_model_settings_allow_environment_override(monkeypatch):
    monkeypatch.setenv("OPENAI_CLASSIFICATION_MODEL", "classification-model")
    monkeypatch.setenv("OPENAI_GENERATION_MODEL", "generation-model")

    configured = Settings(OPENAI_API_KEY="synthetic-key", _env_file=None)

    assert configured.OPENAI_CLASSIFICATION_MODEL == "classification-model"
    assert configured.OPENAI_GENERATION_MODEL == "generation-model"


def test_transaction_classification_uses_classification_model(monkeypatch):
    _FakeChatOpenAI.calls.clear()
    monkeypatch.setattr(llm_service_module, "ChatOpenAI", _FakeChatOpenAI)
    monkeypatch.setattr(
        llm_service_module.settings,
        "OPENAI_CLASSIFICATION_MODEL",
        "gpt-5-nano",
    )

    service = llm_service_module.LLMService(api_key="synthetic-key")

    assert service.model_name == "gpt-5-nano"
    assert _FakeChatOpenAI.calls[0]["model"] == "gpt-5-nano"


def test_api_transaction_classifier_receives_classification_llm_service():
    assert (
        router_module.transaction_classification_service.llm_service
        is llm_service_module.llm_service
    )


def test_recommendation_text_uses_generation_model(monkeypatch):
    _FakeChatOpenAI.calls.clear()
    monkeypatch.setattr(
        recommendation_service_module,
        "ChatOpenAI",
        _FakeChatOpenAI,
    )
    monkeypatch.setattr(
        recommendation_service_module.settings,
        "OPENAI_GENERATION_MODEL",
        "gpt-4o-mini",
    )

    recommendation_service_module.RecommendationService(chroma=object())

    assert _FakeChatOpenAI.calls[0]["model"] == "gpt-4o-mini"


def test_classification_failure_does_not_log_exception_details(
    monkeypatch,
    caplog,
):
    class _FailingChain:
        async def ainvoke(self, payload):
            raise RuntimeError("secret-auth-value")

    class _Prompt:
        def __or__(self, other):
            return _FailingChain()

    service = object.__new__(llm_service_module.LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: _Prompt(),
    )
    transaction = TransactionForClassification(
        transactionId=1,
        amount=1000,
        transactionCategory="ORDINARY",
        organizationCode="0001",
        desc1="민감한가맹점",
    )

    with caplog.at_level(logging.ERROR):
        outcome = asyncio.run(service.classify_with_llm([transaction]))

    assert outcome.results[0].transactionId == 1
    assert "secret-auth-value" not in caplog.text
    assert "민감한가맹점" not in caplog.text


def test_classification_success_logs_counts_without_sensitive_data(
    monkeypatch,
    caplog,
):
    class _SuccessChain:
        async def ainvoke(self, payload):
            return LLMTransactionClassificationResponse(
                results=[
                    TransactionClassificationResult(
                        transactionId=1,
                        isConsumption=True,
                        category=ExpenseCategory.FOOD,
                        expenseType=ExpenseType.VARIABLE,
                    )
                ]
            )

    class _Prompt:
        def __or__(self, other):
            return _SuccessChain()

    service = object.__new__(llm_service_module.LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: _Prompt(),
    )
    transaction = TransactionForClassification(
        transactionId=1,
        amount=1000,
        transactionCategory="ORDINARY",
        organizationCode="0001",
        desc1="민감한가맹점",
    )

    with caplog.at_level(logging.INFO):
        outcome = asyncio.run(service.classify_with_llm([transaction]))

    assert outcome.success is True
    assert "[거래 분류 LLM] 호출 완료" in caplog.text
    assert "llmTargetCount=1" in caplog.text
    assert "rawResultCount=1" in caplog.text
    assert "민감한가맹점" not in caplog.text


def test_recommendation_failure_keeps_deterministic_fallback_and_safe_log(
    caplog,
):
    class _FailingLLM:
        async def ainvoke(self, prompt):
            raise RuntimeError("secret-auth-value")

    service = recommendation_service_module.RecommendationService(
        llm=_FailingLLM(),
        chroma=object(),
    )
    request = ProductRecommendationRequest(
        user_id=1,
        year_month="2026-08",
        total_income=3_000_000,
        total_expense=2_000_000,
        available_funds=1_000_000,
    )
    product = FinancialProductSchema(
        product_id="synthetic-product",
        product_name="synthetic product",
        product_type="SAVINGS",
        provider="synthetic provider",
        summary="synthetic summary",
        details={},
    )

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(
            service._generate_llm_insights(
                request=request,
                product=product,
                simulated_extra_income=1000,
                financial_type="stable",
                financial_activity_insight="summary",
                job_insight="job summary",
                category_expenses=[],
            )
        )

    assert result == {}
    assert "secret-auth-value" not in caplog.text
