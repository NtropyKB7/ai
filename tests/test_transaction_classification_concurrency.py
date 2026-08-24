import asyncio
import json

import pytest

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    LLMTransactionClassificationResponse,
    TransactionCategory,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services import llm_service as llm_service_module
from app.services.llm_service import (
    LLMChunkOutcome,
    LLMClassificationOutcome,
    LLMService,
)
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


def _reset_global_semaphore():
    LLMService._global_classification_semaphore = None
    LLMService._global_classification_semaphore_limit = None


@pytest.fixture(autouse=True)
def _cleanup_global_semaphore():
    _reset_global_semaphore()
    yield
    _reset_global_semaphore()


def _transaction(
    transaction_id: int,
    desc3: str = "알 수 없는 가맹점 XYZ123",
) -> TransactionForClassification:
    return TransactionForClassification(
        transactionId=transaction_id,
        amount=15_000,
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


def _make_transactions(
    count: int,
    start_id: int = 1,
) -> list[TransactionForClassification]:
    return [_transaction(start_id + index) for index in range(count)]


class _Prompt:
    def __init__(self, chain):
        self._chain = chain

    def __or__(self, other):
        return self._chain


class _EchoChain:
    def __init__(self):
        self.calls: list[list[int]] = []

    async def ainvoke(self, payload):
        transactions = json.loads(payload["transactions_json"])
        transaction_ids = [item["transactionId"] for item in transactions]
        self.calls.append(transaction_ids)
        return LLMTransactionClassificationResponse(
            results=[_valid_result(transaction_id) for transaction_id in transaction_ids]
        )


class _BlockingChain:
    def __init__(self, expected_active_calls: int):
        self.expected_active_calls = expected_active_calls
        self.active_calls = 0
        self.max_active_calls = 0
        self.calls: list[list[int]] = []
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, payload):
        transactions = json.loads(payload["transactions_json"])
        transaction_ids = [item["transactionId"] for item in transactions]
        self.calls.append(transaction_ids)

        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        if self.max_active_calls >= self.expected_active_calls:
            self.ready.set()

        try:
            await self.release.wait()
            return LLMTransactionClassificationResponse(
                results=[_valid_result(transaction_id) for transaction_id in transaction_ids]
            )
        finally:
            self.active_calls -= 1


class _TimeoutAwareChain:
    async def ainvoke(self, payload):
        transactions = json.loads(payload["transactions_json"])
        transaction_ids = [item["transactionId"] for item in transactions]
        if transaction_ids[0] == 1:
            await asyncio.sleep(0.05)

        return LLMTransactionClassificationResponse(
            results=[_valid_result(transaction_id) for transaction_id in transaction_ids]
        )


class _InvalidChunkChain:
    async def ainvoke(self, payload):
        transactions = json.loads(payload["transactions_json"])
        transaction_ids = [item["transactionId"] for item in transactions]

        if transaction_ids[0] == 1:
            results = [_valid_result(transaction_id) for transaction_id in transaction_ids]
            results[-1] = _valid_result(transaction_ids[0])
            return LLMTransactionClassificationResponse(results=results)

        return LLMTransactionClassificationResponse(
            results=[_valid_result(transaction_id) for transaction_id in transaction_ids]
        )


def _build_llm_service(
    chain,
    monkeypatch,
    *,
    chunk_size: int = 20,
    concurrency: int = 4,
    timeout_seconds: float = 5.0,
) -> LLMService:
    service = object.__new__(LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    service._classification_chunk_size = chunk_size
    service._classification_max_concurrency = concurrency
    service._classification_timeout_seconds = timeout_seconds
    service._classification_http_timeout_seconds = 25.0
    service._classification_max_retries = 0
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: _Prompt(chain),
    )
    return service


def test_llm_chunks_transactions_into_expected_sizes(monkeypatch):
    chain = _EchoChain()
    service = _build_llm_service(
        chain,
        monkeypatch,
        chunk_size=20,
        concurrency=1,
    )

    outcome = asyncio.run(service.classify_with_llm(_make_transactions(73)))

    assert len(chain.calls) == 4
    assert sorted(len(call) for call in chain.calls) == [13, 20, 20, 20]
    assert outcome.chunk_count == 4
    assert outcome.result_count == 73
    assert outcome.fallback_count == 0
    assert outcome.success is True
    assert [result.transactionId for result in outcome.results] == list(
        range(1, 74)
    )


@pytest.mark.parametrize("concurrency", [2, 4])
def test_llm_chunk_calls_respect_global_concurrency_limit(
    concurrency,
    monkeypatch,
):
    chain = _BlockingChain(expected_active_calls=concurrency)
    service = _build_llm_service(
        chain,
        monkeypatch,
        chunk_size=5,
        concurrency=concurrency,
        timeout_seconds=5.0,
    )

    async def run_test():
        task = asyncio.create_task(
            service.classify_with_llm(_make_transactions(40))
        )
        await asyncio.wait_for(chain.ready.wait(), timeout=2.0)
        assert chain.max_active_calls == concurrency
        chain.release.set()
        return await task

    outcome = asyncio.run(run_test())

    assert outcome.chunk_count == 8
    assert outcome.result_count == 40
    assert outcome.fallback_count == 0
    assert outcome.success is True
    assert chain.max_active_calls == concurrency


def test_llm_timeout_falls_back_only_for_timed_out_chunk(
    caplog,
    monkeypatch,
):
    chain = _TimeoutAwareChain()
    service = _build_llm_service(
        chain,
        monkeypatch,
        chunk_size=20,
        concurrency=2,
        timeout_seconds=0.01,
    )

    with caplog.at_level("ERROR"):
        outcome = asyncio.run(service.classify_with_llm(_make_transactions(40)))

    assert outcome.chunk_count == 2
    assert outcome.result_count == 20
    assert outcome.fallback_count == 20
    assert outcome.success is False
    assert outcome.chunk_outcomes[0].success is False
    assert outcome.chunk_outcomes[0].fallback_count == 20
    assert outcome.chunk_outcomes[1].success is True
    assert outcome.chunk_outcomes[1].fallback_count == 0
    assert "민감한가맹점" not in caplog.text
    assert "TimeoutError" in caplog.text or "OpenAITimeoutError" in caplog.text


def test_llm_invalid_chunk_falls_back_only_for_that_chunk(
    caplog,
    monkeypatch,
):
    chain = _InvalidChunkChain()
    service = _build_llm_service(
        chain,
        monkeypatch,
        chunk_size=20,
        concurrency=2,
        timeout_seconds=5.0,
    )
    transactions = _make_transactions(40)
    transactions[0] = _transaction(1, desc3="민감한가맹점")

    with caplog.at_level("ERROR"):
        outcome = asyncio.run(service.classify_with_llm(transactions))

    assert outcome.chunk_count == 2
    assert outcome.result_count == 20
    assert outcome.fallback_count == 20
    assert outcome.success is False
    assert outcome.chunk_outcomes[0].success is False
    assert outcome.chunk_outcomes[0].fallback_count == 20
    assert outcome.chunk_outcomes[1].success is True
    assert outcome.chunk_outcomes[1].fallback_count == 0
    assert "민감한가맹점" not in caplog.text


def test_final_classification_log_records_chunk_and_fallback_counts(
    caplog,
):
    class _FakeLLMService:
        async def classify_with_llm(self, transactions):
            return LLMClassificationOutcome(
                results=[_valid_result(transactions[0].transactionId)],
                llm_target_count=len(transactions),
                result_count=1,
                raw_result_count=1,
                chunk_count=2,
                elapsed_ms=42,
                success=False,
                error_kind="TimeoutError",
                fallback_count=1,
                chunk_outcomes=(
                    LLMChunkOutcome(
                        chunk_index=1,
                        chunk_size=1,
                        results=[_valid_result(transactions[0].transactionId)],
                        result_count=1,
                        raw_result_count=1,
                        elapsed_ms=10,
                        success=True,
                        error_kind=None,
                        fallback_count=0,
                    ),
                    LLMChunkOutcome(
                        chunk_index=2,
                        chunk_size=1,
                        results=[],
                        result_count=0,
                        raw_result_count=0,
                        elapsed_ms=32,
                        success=False,
                        error_kind="TimeoutError",
                        fallback_count=1,
                    ),
                ),
            )

    service = TransactionClassificationService(llm_service=_FakeLLMService())
    transactions = [
        _transaction(1, desc3="스타벅스 강남점"),
        _transaction(2, desc3="민감한가맹점"),
        _transaction(3, desc3="알 수 없는 가맹점"),
    ]

    with caplog.at_level("INFO"):
        results = asyncio.run(service.classify_transactions_async(transactions))

    assert len(results) == 3
    assert [result.transactionId for result in results] == [1, 2, 3]
    assert "[거래 분류 완료]" in caplog.text
    assert "transactionCount=3" in caplog.text
    assert "ruleCount=1" in caplog.text
    assert "llmTargetCount=2" in caplog.text
    assert "chunkCount=2" in caplog.text
    assert "resultCount=1" in caplog.text
    assert "fallbackCount=1" in caplog.text
    assert "totalElapsedMs=" in caplog.text
    assert "민감한가맹점" not in caplog.text
