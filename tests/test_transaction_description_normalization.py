import json
import asyncio
from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    LLMTransactionClassificationResponse,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services import llm_service as llm_service_module
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)
from app.services.transaction_description_normalizer import (
    normalize_transaction_description,
)


def transaction(desc3: str) -> TransactionForClassification:
    return TransactionForClassification(
        transactionId=1,
        amount=10_000,
        transactionCategory="ORDINARY",
        organizationCode="0004",
        desc1=None,
        desc2="전자금융",
        desc3=desc3,
        desc4=None,
    )


@pytest.mark.parametrize(
    ("desc3", "payment_method", "merchant_candidate"),
    [
        ("카카오페이_공차", "카카오페이", "공차"),
        ("토스페이_29CM", "토스페이", "29CM"),
        ("인터파크 티켓", None, "인터파크 티켓"),
        ("_공차", None, "_공차"),
        ("카카오페이_", None, "카카오페이_"),
        ("결제수단_가맹점_추가정보", None, "결제수단_가맹점_추가정보"),
    ],
)
def test_desc3_normalization(desc3, payment_method, merchant_candidate):
    context = normalize_transaction_description(transaction(desc3))

    assert context.original_description == desc3
    assert context.payment_method == payment_method
    assert context.merchant_candidate == merchant_candidate


@pytest.mark.parametrize(
    ("desc3", "category"),
    [
        ("카카오페이_공차", ExpenseCategory.FOOD),
        ("토스페이_29CM", ExpenseCategory.SHOPPING),
        ("카카오페이_올리브영", ExpenseCategory.SHOPPING),
        ("공방원데이클래스", ExpenseCategory.LEISURE),
        ("인터파크 티켓", ExpenseCategory.LEISURE),
    ],
)
def test_normalized_merchant_rule_classification_skips_llm(desc3, category):
    llm = Mock()
    service = TransactionClassificationService(llm_service=llm)

    original = transaction(desc3)
    original_copy = deepcopy(original)
    result = service.classify_transactions([original])[0]

    assert result.category == category
    assert result.expenseType == ExpenseType.VARIABLE
    assert original == original_copy
    llm.classify_with_llm.assert_not_called()


def test_rule_miss_sends_original_transaction_to_llm_once():
    llm = Mock()
    llm.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=1,
            isConsumption=True,
            category=ExpenseCategory.LEISURE,
            expenseType=ExpenseType.VARIABLE,
        )
    ]
    service = TransactionClassificationService(llm_service=llm)
    original = transaction("간편결제_새로운가맹점")

    result = service.classify_transactions([original])[0]

    assert result.category == ExpenseCategory.LEISURE
    llm.classify_with_llm.assert_called_once_with([original])


def test_llm_receives_original_and_normalized_description(monkeypatch):
    captured = {}

    class Chain:
        async def ainvoke(self, values):
            captured.update(values)
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

    class Prompt:
        def __or__(self, llm):
            return Chain()

    service = object.__new__(llm_service_module.LLMService)
    service.llm = object()
    service.system_prompt = "classification prompt"
    monkeypatch.setattr(
        llm_service_module.ChatPromptTemplate,
        "from_messages",
        lambda messages: Prompt(),
    )

    outcome = asyncio.run(service.classify_with_llm([transaction("카카오페이_공차")]))
    payload = json.loads(captured["transactions_json"])[0]

    assert outcome.results[0].category == ExpenseCategory.FOOD
    assert payload["originalDescription"] == "카카오페이_공차"
    assert payload["paymentMethod"] == "카카오페이"
    assert payload["merchantCandidate"] == "공차"


def test_llm_exception_isolated_by_service_fallback():
    llm = Mock()
    llm.classify_with_llm.side_effect = RuntimeError("sensitive detail")
    service = TransactionClassificationService(llm_service=llm)

    result = service.classify_transactions([transaction("알수없는거래")])[0]

    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE
