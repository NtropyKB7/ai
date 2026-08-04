# datetime은 테스트 거래의 거래 일시를 만들기 위해 사용합니다.
from datetime import datetime

# pytest는 같은 형태의 테스트를 여러 번 실행하는 parametrize에 사용합니다.
import pytest

# 실제 서비스가 사용하는 DTO와 Enum을 가져옵니다.
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionForClassification,
    TransactionType,
)

# 테스트할 규칙 기반 분류 서비스를 가져옵니다.
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


# 모든 테스트에서 재사용할 서비스 객체입니다.
service = TransactionClassificationService()


def create_transaction(**overrides) -> TransactionForClassification:
    """
    기본 거래 데이터를 만들고, 테스트마다 필요한 값만 덮어씁니다.

    예를 들어 create_transaction(txnType=TransactionType.IN)처럼 사용합니다.
    """

    transaction_data = {
        "transactionId": 1001,
        "transactionDate": datetime(2026, 7, 18, 14, 32, 10),
        "txnType": TransactionType.OUT,
        "amount": 15000,
        "merchantName": "테스트 거래처",
        "transactionDetails": "테스트 거래 내용",
        "jobId": None,
        "jobName": None,
    }

    # 테스트에서 전달한 값으로 기본값을 변경합니다.
    transaction_data.update(overrides)

    return TransactionForClassification(**transaction_data)


def test_in_transaction_is_not_consumption():
    """
    입금(IN) 거래는 소비 거래가 아닌지 확인합니다.
    """

    transaction = create_transaction(
        txnType=TransactionType.IN,
        merchantName="급여",
        transactionDetails="급여 입금",
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is False
    assert result.category is None
    assert result.expenseType is None
    assert result.confidence == 0.99


@pytest.mark.parametrize(
    ("merchant_name", "transaction_details"),
    [
        ("카카오뱅크", "타행 이체"),
        ("대출 상환", "원금 상환"),
    ],
)
def test_non_consumption_keywords_are_not_consumption(
    merchant_name: str,
    transaction_details: str,
):
    """
    타행이체와 대출 상환처럼 비소비 키워드가 포함된 출금은
    소비 거래가 아닌지 확인합니다.
    """

    transaction = create_transaction(
        merchantName=merchant_name,
        transactionDetails=transaction_details,
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is False
    assert result.category is None
    assert result.expenseType is None
    assert result.confidence == 0.95


@pytest.mark.parametrize(
    "merchant_name",
    [
        "스타벅스",
        "배달의민족",
    ],
)
def test_food_transactions_are_classified_as_food(merchant_name: str):
    """
    스타벅스와 배달 거래가 식비 / 변동 지출로 분류되는지 확인합니다.
    """

    transaction = create_transaction(
        merchantName=merchant_name,
        transactionDetails="카드결제",
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.FOOD
    assert result.expenseType == ExpenseType.VARIABLE
    assert result.confidence == 0.95


def test_subscription_is_classified_as_fixed_expense():
    """
    넷플릭스 정기결제 거래가 정기구독 / 고정 지출로 분류되는지 확인합니다.
    """

    transaction = create_transaction(
        merchantName="넷플릭스",
        transactionDetails="정기결제",
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.SUBSCRIPTION
    assert result.expenseType == ExpenseType.FIXED
    assert result.confidence == 0.95


@pytest.mark.parametrize(
    ("merchant_name", "expected_category"),
    [
        ("아파트 관리비", ExpenseCategory.HOUSING),
        ("삼성화재 보험료", ExpenseCategory.FINANCE),
    ],
)
def test_fixed_expenses_are_classified_as_fixed(
    merchant_name: str,
    expected_category: ExpenseCategory,
):
    """
    관리비와 보험료가 각각 올바른 카테고리이며,
    고정 지출(FIXED)로 분류되는지 확인합니다.
    """

    transaction = create_transaction(
        merchantName=merchant_name,
        transactionDetails="자동납부",
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is True
    assert result.category == expected_category
    assert result.expenseType == ExpenseType.FIXED
    assert result.confidence == 0.95


def test_unknown_out_transaction_is_classified_as_other():
    """
    어떤 키워드에도 일치하지 않는 출금 거래는
    OTHER / VARIABLE / 낮은 신뢰도로 반환되는지 확인합니다.
    """

    transaction = create_transaction(
        merchantName="알수없는상점",
        transactionDetails="일반 출금",
    )

    result = service.classify_transaction(transaction)

    assert result.isConsumption is True
    assert result.category == ExpenseCategory.OTHER
    assert result.expenseType == ExpenseType.VARIABLE
    assert result.confidence == 0.40


def test_bulk_classification_returns_all_results():
    """
    여러 거래를 전달했을 때 거래 수만큼 분류 결과가 반환되는지 확인합니다.
    """

    transactions = [
        create_transaction(
            transactionId=1001,
            merchantName="스타벅스",
        ),
        create_transaction(
            transactionId=1002,
            txnType=TransactionType.IN,
            merchantName="급여",
        ),
    ]

    results = service.classify_transactions(transactions)

    assert len(results) == 2
    assert results[0].transactionId == 1001
    assert results[1].transactionId == 1002