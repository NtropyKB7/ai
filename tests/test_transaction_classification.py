from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionCategory,
    TransactionClassificationRequest,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)

client = TestClient(app)


def create_transaction(
    transaction_id: int = 1,
    amount: int = 10_000,
    organization_code: str = "0004",
    desc1: str | None = None,
    desc2: str | None = None,
    desc3: str | None = None,
    desc4: str | None = None,
) -> TransactionForClassification:
    return TransactionForClassification(
        transactionId=transaction_id,
        amount=amount,
        transactionCategory=TransactionCategory.ORDINARY,
        organizationCode=organization_code,
        desc1=desc1,
        desc2=desc2,
        desc3=desc3,
        desc4=desc4,
    )


def test_request_accepts_structured_transaction_fields():
    request = TransactionClassificationRequest(
        transactions=[
            create_transaction(
                transaction_id=101,
                amount=65_000,
                organization_code="0004",
                desc2="FBS출금",
                desc3="KT통신요금",
                desc4="강남지점",
            )
        ]
    )

    transaction = request.transactions[0]

    assert transaction.transactionId == 101
    assert transaction.amount == 65_000
    assert transaction.transactionCategory == TransactionCategory.ORDINARY
    assert transaction.organizationCode == "0004"
    assert transaction.desc1 is None
    assert transaction.desc2 == "FBS출금"
    assert transaction.desc3 == "KT통신요금"
    assert transaction.desc4 == "강남지점"


def test_request_rejects_zero_or_negative_amount():
    with pytest.raises(ValidationError):
        create_transaction(amount=0)

    with pytest.raises(ValidationError):
        create_transaction(amount=-1)


def test_request_rejects_non_positive_transaction_id():
    with pytest.raises(ValidationError):
        create_transaction(transaction_id=0)


def test_request_rejects_more_than_100_transactions():
    transactions = [
        create_transaction(transaction_id=index)
        for index in range(1, 102)
    ]

    with pytest.raises(ValidationError):
        TransactionClassificationRequest(transactions=transactions)


def test_request_rejects_duplicate_transaction_ids():
    with pytest.raises(
        ValidationError,
        match="transactionId must be unique",
    ):
        TransactionClassificationRequest(
            transactions=[
                create_transaction(transaction_id=1),
                create_transaction(transaction_id=1),
            ]
        )


def test_all_description_fields_may_be_null():
    transaction = create_transaction(
        desc1=None,
        desc2=None,
        desc3=None,
        desc4=None,
    )

    assert transaction.desc1 is None
    assert transaction.desc2 is None
    assert transaction.desc3 is None
    assert transaction.desc4 is None


@pytest.mark.parametrize(
    ("desc3", "expected_category", "expected_expense_type"),
    [
        ("스타벅스 강남점", ExpenseCategory.FOOD, ExpenseType.VARIABLE),
        (
            "카카오T 택시",
            ExpenseCategory.TRANSPORTATION,
            ExpenseType.VARIABLE,
        ),
        ("한국전력", ExpenseCategory.HOUSING, ExpenseType.FIXED),
        ("KT통신요금", ExpenseCategory.COMMUNICATION, ExpenseType.FIXED),
        ("서울대학교병원", ExpenseCategory.MEDICAL, ExpenseType.VARIABLE),
        ("패스트캠퍼스 수강료", ExpenseCategory.EDUCATION, ExpenseType.FIXED),
        ("신세계백화점", ExpenseCategory.SHOPPING, ExpenseType.VARIABLE),
        ("넷플릭스", ExpenseCategory.LEISURE, ExpenseType.VARIABLE),
    ],
)
def test_rule_based_category_classification(
    desc3,
    expected_category,
    expected_expense_type,
):
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc3=desc3)]
    )[0]

    assert result.isConsumption is True
    assert result.category == expected_category
    assert result.expenseType == expected_expense_type


def test_desc3_is_used_as_primary_merchant_field():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [
            create_transaction(
                desc2="FBS출금",
                desc3="KT통신요금",
                desc4="강남지점",
            )
        ]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.COMMUNICATION
    assert result.expenseType == ExpenseType.FIXED


@pytest.mark.parametrize(
    "description",
    [
        "정기적금",
        "자유적금",
        "적금납입",
        "정기예금",
        "대출상환",
        "원금상환",
        "대출계좌",
        "본인계좌",
    ],
)
def test_explicit_financial_account_transfer_is_non_consumption(
    description,
):
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc3=description)]
    )[0]

    assert result.isConsumption is False
    assert result.category is None
    assert result.expenseType is None


@pytest.mark.parametrize(
    "transaction_channel",
    [
        "자동이체",
        "타행이체",
        "전자금융",
        "CMS",
        "FBS출금",
        "인터넷뱅킹",
    ],
)
def test_generic_transaction_channel_alone_is_not_non_consumption(
    transaction_channel,
):
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc2=transaction_channel)]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


def test_insurance_is_fixed_consumption():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [
            create_transaction(
                desc2="자동이체",
                desc3="삼성생명 실손보험",
            )
        ]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.INSURANCE
    assert result.expenseType == ExpenseType.FIXED


def test_loan_interest_is_fixed_finance_consumption():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc3="대출이자 납입")]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.FINANCE
    assert result.expenseType == ExpenseType.FIXED


def test_card_bill_is_variable_etc_consumption():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc3="신용카드이용대금")]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


@pytest.mark.parametrize(
    "description",
    [
        "ATM출금",
        "스마트출금",
        "현금인출",
        "CD출금",
    ],
)
def test_cash_withdrawal_is_variable_etc_consumption(description):
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc2=description)]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


def test_one_time_financial_fee_is_variable_finance():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [create_transaction(desc3="해외결제수수료")]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.FINANCE
    assert result.expenseType == ExpenseType.VARIABLE


def test_check_card_marker_does_not_hide_merchant_classification():
    service = TransactionClassificationService()

    result = service.classify_transactions(
        [
            create_transaction(
                desc2="NHBC체크",
                desc3="스타벅스",
            )
        ]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.FOOD
    assert result.expenseType == ExpenseType.VARIABLE


def test_unclassified_transaction_is_sent_to_llm():
    llm_service = Mock()
    llm_service.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=1,
            isConsumption=True,
            category=ExpenseCategory.SHOPPING,
            expenseType=ExpenseType.VARIABLE,
        )
    ]

    service = TransactionClassificationService(llm_service=llm_service)
    transaction = create_transaction(desc3="알 수 없는 가맹점")

    result = service.classify_transactions([transaction])[0]

    llm_service.classify_with_llm.assert_called_once_with([transaction])
    assert result.category == ExpenseCategory.SHOPPING


def test_rule_result_is_not_overridden_by_llm():
    llm_service = Mock()
    llm_service.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=1,
            isConsumption=True,
            category=ExpenseCategory.SHOPPING,
            expenseType=ExpenseType.VARIABLE,
        )
    ]

    service = TransactionClassificationService(llm_service=llm_service)

    result = service.classify_transactions(
        [create_transaction(desc3="스타벅스")]
    )[0]

    llm_service.classify_with_llm.assert_not_called()
    assert result.category == ExpenseCategory.FOOD


def test_missing_llm_result_uses_fallback():
    llm_service = Mock()
    llm_service.classify_with_llm.return_value = []

    service = TransactionClassificationService(llm_service=llm_service)

    result = service.classify_transactions(
        [create_transaction(desc3="알 수 없는 거래")]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


def test_duplicate_llm_results_use_fallback():
    llm_service = Mock()
    duplicated_result = TransactionClassificationResult(
        transactionId=1,
        isConsumption=True,
        category=ExpenseCategory.FOOD,
        expenseType=ExpenseType.VARIABLE,
    )
    llm_service.classify_with_llm.return_value = [
        duplicated_result,
        duplicated_result,
    ]

    service = TransactionClassificationService(llm_service=llm_service)

    result = service.classify_transactions(
        [create_transaction(desc3="알 수 없는 거래")]
    )[0]

    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


def test_unknown_llm_transaction_id_is_ignored():
    llm_service = Mock()
    llm_service.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=999,
            isConsumption=True,
            category=ExpenseCategory.FOOD,
            expenseType=ExpenseType.VARIABLE,
        )
    ]

    service = TransactionClassificationService(llm_service=llm_service)

    result = service.classify_transactions(
        [create_transaction(transaction_id=1, desc3="알 수 없는 거래")]
    )[0]

    assert result.transactionId == 1
    assert result.category == ExpenseCategory.ETC


def test_invalid_non_consumption_llm_result_uses_fallback():
    llm_service = Mock()
    llm_service.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=1,
            isConsumption=False,
            category=ExpenseCategory.FOOD,
            expenseType=ExpenseType.VARIABLE,
        )
    ]

    service = TransactionClassificationService(llm_service=llm_service)

    result = service.classify_transactions(
        [create_transaction(desc3="알 수 없는 거래")]
    )[0]

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.ETC
    assert result.expenseType == ExpenseType.VARIABLE


def test_api_classifies_structured_request():
    response = client.post(
        "/api/v1/classify-transactions",
        json={
            "transactions": [
                {
                    "transactionId": 101,
                    "amount": 15_000,
                    "transactionCategory": "ORDINARY",
                    "organizationCode": "0004",
                    "desc1": None,
                    "desc2": "FBS출금",
                    "desc3": "스타벅스 강남점",
                    "desc4": "강남지점",
                }
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()
    result = body["data"]["results"][0]

    assert result == {
        "transactionId": 101,
        "isConsumption": True,
        "category": "FOOD",
        "expenseType": "VARIABLE",
    }