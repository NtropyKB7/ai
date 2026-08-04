# pytest는 "예외가 발생해야 하는 테스트"를 작성할 때 사용합니다.
import pytest

# Pydantic DTO 검증 오류의 타입입니다.
from pydantic import ValidationError

# 검증할 요청 DTO와 거래 DTO를 가져옵니다.
from app.schemas.transaction import (
    TransactionClassificationRequest,
    TransactionForClassification,
)


def create_valid_transaction_data() -> dict:
    """
    유효한 거래 데이터의 기본 형태를 반환합니다.

    테스트마다 필요한 필드만 수정해서 사용할 수 있습니다.
    """

    return {
        "transactionId": 1001,
        "transactionDate": "2026-07-18T14:32:10",
        "txnType": "OUT",
        "amount": 15000,
        "merchantName": "스타벅스",
        "transactionDetails": "카드결제",
        "jobId": None,
        "jobName": None,
    }


def test_negative_amount_is_rejected():
    """
    거래 금액이 음수이면 DTO 검증 오류가 발생하는지 확인합니다.
    """

    transaction_data = create_valid_transaction_data()

    # amount 필드는 0 이상만 허용하므로 음수로 변경합니다.
    transaction_data["amount"] = -1000

    with pytest.raises(ValidationError):
        TransactionForClassification(**transaction_data)


def test_empty_transaction_list_is_rejected():
    """
    분류할 거래 목록이 비어 있으면 DTO 검증 오류가 발생하는지 확인합니다.
    """

    with pytest.raises(ValidationError):
        TransactionClassificationRequest(transactions=[])


def test_invalid_transaction_type_is_rejected():
    """
    txnType에는 IN 또는 OUT만 허용되므로,
    다른 값이 들어오면 검증 오류가 발생하는지 확인합니다.
    """

    transaction_data = create_valid_transaction_data()

    # 계약에 없는 입출금 구분값입니다.
    transaction_data["txnType"] = "PAYMENT"

    with pytest.raises(ValidationError):
        TransactionForClassification(**transaction_data)