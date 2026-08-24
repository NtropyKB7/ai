import asyncio
from typing import Any

import httpx

from app.api import router as router_module
from app.main import app
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    LLMTransactionClassificationResponse,
    TransactionCategory,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services import llm_service as llm_service_module
from app.services.llm_service import LLMClassificationOutcome, LLMService
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


def _transaction(
    transaction_id: int = 1,
    desc3: str = "알 수 없는 가맹점",
) -> TransactionForClassification:
    return TransactionForClassification(
        transactionId=transaction_id,
        amount=10_000,
        transactionCategory=TransactionCategory.ORDINARY,
        organizationCode="0004",
        desc1=None,
        desc2="전자금융",
        desc3=desc3,
        desc4=None,
    )


def _valid_result(transaction_id: int) -> TransactionClassificationResult:
    return TransactionClassificationResult(
        transactionId=transaction_id,
        isConsumption=True,
        category=ExpenseCategory.FOOD,
        expenseType=ExpenseType.VARIABLE,
    )


class _OverlapLLMService:
    def __init__(self):
        self.active_calls = 0
        self.max_active_calls = 0
        self.overlap_observed = asyncio.Event()
        self.release = asyncio.Event()

    async def classify_with_llm(
        self,
        transactions: list[TransactionForClassification],
    ) -> LLMClassificationOutcome:
        self.active_calls += 1
        self.max_active_calls = max(
            self.max_active_calls,
            self.active_calls,
        )
        if self.max_active_calls >= 2:
            self.overlap_observed.set()

        try:
            await self.release.wait()
            return LLMClassificationOutcome(
                results=[_valid_result(item.transactionId) for item in transactions],
                llm_target_count=len(transactions),
                raw_result_count=len(transactions),
                elapsed_ms=0,
                success=True,
                error_kind=None,
                fallback_count=0,
            )
        finally:
            self.active_calls -= 1


class _BlockingChain:
    def __init__(self, response: Any):
        self.response = response
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, payload: dict[str, Any]):
        self.calls += 1
        self.active_calls += 1
        self.max_active_calls = max(
            self.max_active_calls,
            self.active_calls,
        )
        self.entered.set()
        try:
            await self.release.wait()
            return self.response
        finally:
            self.active_calls -= 1


async def _request_classification(client: httpx.AsyncClient, transaction_id: int):
    response = await client.post(
        "/api/v1/classify-transactions",
        json={
            "transactions": [
                {
                    "transactionId": transaction_id,
                    "amount": 15_000,
                    "transactionCategory": "ORDINARY",
                    "organizationCode": "0004",
                    "desc1": None,
                    "desc2": "전자금융",
                    "desc3": "알 수 없는 가맹점",
                    "desc4": None,
                }
            ]
        },
    )
    return response


def test_http_requests_can_overlap_in_fastapi(monkeypatch):
    fake_llm = _OverlapLLMService()
    monkeypatch.setattr(
        router_module.transaction_classification_service,
        "llm_service",
        fake_llm,
    )

    async def run_test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            task1 = asyncio.create_task(_request_classification(client, 101))
            await asyncio.sleep(0)
            task2 = asyncio.create_task(_request_classification(client, 102))

            await asyncio.wait_for(fake_llm.overlap_observed.wait(), timeout=2.0)
            fake_llm.release.set()

            return await asyncio.gather(task1, task2)

    responses = asyncio.run(run_test())

    assert all(response.status_code == 200 for response in responses)
    assert fake_llm.max_active_calls >= 2


def test_llm_semaphore_limits_concurrent_calls(monkeypatch):
    response = LLMTransactionClassificationResponse(
        results=[_valid_result(1)]
    )
    fake_chain = _BlockingChain(response=response)

    class _Prompt:
        def __or__(self, other):
            return fake_chain

    service = object.__new__(LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    service._classification_semaphore = asyncio.Semaphore(1)
    service._classification_timeout_seconds = 5.0
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: _Prompt(),
    )

    async def run_test():
        task1 = asyncio.create_task(
            service.classify_with_llm([_transaction(1)])
        )
        await asyncio.wait_for(fake_chain.entered.wait(), timeout=2.0)

        task2 = asyncio.create_task(
            service.classify_with_llm([_transaction(2)])
        )
        await asyncio.sleep(0)

        assert fake_chain.calls == 1
        assert fake_chain.max_active_calls == 1

        fake_chain.release.set()
        outcome1, outcome2 = await asyncio.gather(task1, task2)
        return outcome1, outcome2

    outcome1, outcome2 = asyncio.run(run_test())

    assert outcome1.success is True
    assert outcome2.success is True
    assert fake_chain.calls == 2
    assert fake_chain.max_active_calls == 1


def test_llm_timeout_falls_back_without_sensitive_logs(monkeypatch, caplog):
    class _SlowChain:
        async def ainvoke(self, payload):
            await asyncio.sleep(0.05)
            return LLMTransactionClassificationResponse(
                results=[_valid_result(1)]
            )

    class _Prompt:
        def __or__(self, other):
            return _SlowChain()

    service = object.__new__(LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    service._classification_semaphore = asyncio.Semaphore(1)
    service._classification_timeout_seconds = 0.01
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: _Prompt(),
    )
    transaction = _transaction(1, desc3="민감한가맹점")

    with caplog.at_level("ERROR"):
        outcome = asyncio.run(service.classify_with_llm([transaction]))

    assert outcome.success is False
    assert outcome.error_kind == "TimeoutError"
    assert outcome.fallback_count == 1
    assert len(outcome.results) == 1
    assert outcome.results[0].category == ExpenseCategory.ETC
    assert outcome.results[0].expenseType == ExpenseType.VARIABLE
    assert "민감한가맹점" not in caplog.text
    assert "classification prompt" not in caplog.text


def test_final_classification_log_records_counts_without_sensitive_data(
    monkeypatch,
    caplog,
):
    class _FakeLLMService:
        async def classify_with_llm(self, transactions):
            return LLMClassificationOutcome(
                results=[_valid_result(transactions[0].transactionId)],
                llm_target_count=len(transactions),
                raw_result_count=len(transactions),
                elapsed_ms=42,
                success=True,
                error_kind=None,
                fallback_count=0,
            )

    service = TransactionClassificationService(llm_service=_FakeLLMService())
    transactions = [
        _transaction(1, desc3="민감한가맹점"),
        TransactionForClassification(
            transactionId=2,
            amount=20_000,
            transactionCategory=TransactionCategory.ORDINARY,
            organizationCode="0004",
            desc1=None,
            desc2=None,
            desc3="스타벅스 강남점",
            desc4=None,
        ),
    ]

    with caplog.at_level("INFO"):
        results = asyncio.run(service.classify_transactions_async(transactions))

    assert len(results) == 2
    assert "[거래 분류 완료]" in caplog.text
    assert "transactionCount=2" in caplog.text
    assert "llmTargetCount=1" in caplog.text
    assert "resultCount=1" in caplog.text
    assert "fallbackCount=0" in caplog.text
    assert "민감한가맹점" not in caplog.text
