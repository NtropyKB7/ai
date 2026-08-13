"""
은행별 desc 필드 해석에 대한 보조 회귀 테스트입니다.
"""

from unittest.mock import MagicMock

import pytest

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionForClassification,
)
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


@pytest.fixture
def service() -> TransactionClassificationService:
    """
    규칙으로 분류되지 않은 거래가 외부 LLM을 실제 호출하지 않도록
    테스트용 LLM 객체를 사용합니다.
    """
    llm_service = MagicMock()
    llm_service.classify_with_llm.return_value = []

    return TransactionClassificationService(
        llm_service=llm_service
    )


@pytest.mark.parametrize(
    (
        "organization_code",
        "desc1",
        "desc2",
        "desc3",
        "desc4",
        "expected_category",
        "expected_expense_type",
    ),
    [
        (
            "0003",
            "스타벅스",
            "체크",
            "스타벅스",
            None,
            ExpenseCategory.FOOD,
            ExpenseType.VARIABLE,
        ),
        (
            "0004",
            None,
            "FBS출금",
            "KT통신요금",
            "강남지점",
            ExpenseCategory.COMMUNICATION,
            ExpenseType.FIXED,
        ),
        (
            "0011",
            None,
            "보험료",
            "삼성생명 실손보험",
            "농협 000911",
            ExpenseCategory.INSURANCE,
            ExpenseType.FIXED,
        ),
        (
            "0023",
            None,
            "C/C",
            "다이소",
            "강남지점",
            ExpenseCategory.SHOPPING,
            ExpenseType.VARIABLE,
        ),
        (
            "0037",
            None,
            None,
            "홈)한국전력",
            "전북(본점)",
            ExpenseCategory.HOUSING,
            ExpenseType.FIXED,
        ),
    ],
)
def test_bank_description_fields(
    service,
    organization_code,
    desc1,
    desc2,
    desc3,
    desc4,
    expected_category,
    expected_expense_type,
):
    """
    은행마다 desc1~4의 저장 형태가 달라도 desc3의 상대방·가맹점 정보를
    중심으로 올바른 소비 카테고리와 지출 유형을 결정해야 합니다.
    """
    transaction = TransactionForClassification(
        transactionId=1,
        amount=10_000,
        transactionCategory="ORDINARY",
        organizationCode=organization_code,
        desc1=desc1,
        desc2=desc2,
        desc3=desc3,
        desc4=desc4,
    )

    result = service.classify_transactions([transaction])[0]

    assert result.isConsumption is True
    assert result.category == expected_category
    assert result.expenseType == expected_expense_type