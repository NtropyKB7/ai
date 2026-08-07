from datetime import datetime
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionClassificationRequest,
    TransactionClassificationResult,
    TransactionForClassification,
)
from app.services.transaction_classification_service import (
    TransactionClassificationService,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def mock_llm_service():
    """LLM 외부 API 호출을 차단하기 위한 Mock Fixture"""
    service = MagicMock()
    service.classify_with_llm.return_value = []
    return service


@pytest.fixture
def classification_service(mock_llm_service):
    """테스트 대상 분류 서비스 인스턴스 Fixture"""
    return TransactionClassificationService(llm_service=mock_llm_service)


def create_txn(
    txn_id: int, description: str, merchant_name: str = "", amount: int = 10000
) -> TransactionForClassification:
    """테스트용 거래 객체 생성 헬퍼 함수"""
    return TransactionForClassification(
        transactionId=txn_id,
        transactionDate=datetime.now(),
        amount=amount,
        merchantName=merchant_name,
        description=description,
    )


# =========================================================================
# 1. DTO 검증 테스트
# =========================================================================

def test_transaction_for_classification_valid():
    """정상적인 DTO 생성 검증"""
    txn = TransactionForClassification(
        transactionId=1,
        transactionDate=datetime(2026, 8, 7, 12, 0, 0),
        amount=15000,
        merchantName="스타벅스 강남점",
        description="스타벅스",
    )
    assert txn.transactionId == 1
    assert txn.amount == 15000
    assert txn.merchantName == "스타벅스 강남점"


def test_transaction_for_classification_negative_amount():
    """음수 금액 입력 시 Pydantic ValidationError 발생 검증"""
    with pytest.raises(ValidationError):
        TransactionForClassification(
            transactionId=1,
            transactionDate=datetime.now(),
            amount=-5000,  # ge=0 검증 위반
            description="잘못된 금액",
        )


# =========================================================================
# 2. 비소비 거래 규칙 테스트
# =========================================================================

@pytest.mark.parametrize(
    "description, merchant",
    [
        ("청약저축 납입", "KB국민은행"),
        ("대출원금 상환", "신한은행"),
        ("내계좌 이체", "홍길동"),
        ("ATM 현금인출", "하나은행 ATM"),
        ("주식 매수", "키움증권"),
    ],
)
def test_rule_non_consumption_transactions(classification_service, description, merchant):
    """적금, 대출원금, 계좌이체, 현금인출 등 비소비 거래 판별 테스트"""
    txn = create_txn(1, description=description, merchant_name=merchant)
    results = classification_service.classify_transactions([txn])

    assert len(results) == 1
    res = results[0]
    assert res.transactionId == 1
    assert res.isConsumption is False
    assert res.category is None
    assert res.expenseType is None


# =========================================================================
# 3. 금융 예외 소비 거래 규칙 테스트
# =========================================================================

def test_rule_financial_exception_loan_interest(classification_service):
    """대출 이자: 소비 / FINANCE / FIXED 검증"""
    txn = create_txn(1, description="주택담보대출이자 납입", merchant_name="우리은행")
    res = classification_service.classify_transactions([txn])[0]

    assert res.isConsumption is True
    assert res.category == ExpenseCategory.FINANCE
    assert res.expenseType == ExpenseType.FIXED


def test_rule_financial_exception_insurance(classification_service):
    """보험료: 소비 / INSURANCE / FIXED 검증"""
    txn = create_txn(1, description="실손보험료 출금", merchant_name="삼성화재")
    res = classification_service.classify_transactions([txn])[0]

    assert res.isConsumption is True
    assert res.category == ExpenseCategory.INSURANCE
    assert res.expenseType == ExpenseType.FIXED


def test_rule_financial_exception_card_bill(classification_service):
    """카드대금: 소비 / ETC / VARIABLE (MVP 예외) 검증"""
    txn = create_txn(1, description="8월 카드대금 결제", merchant_name="현대카드")
    res = classification_service.classify_transactions([txn])[0]

    assert res.isConsumption is True
    assert res.category == ExpenseCategory.ETC
    assert res.expenseType == ExpenseType.VARIABLE


# =========================================================================
# 4. 주요 카테고리 패턴 매칭 테스트
# =========================================================================

@pytest.mark.parametrize(
    "description, merchant, expected_category, expected_type",
    [
        ("배달의민족 결제", "", ExpenseCategory.FOOD, ExpenseType.VARIABLE),
        ("스타벅스 아메리카노", "스타벅스", ExpenseCategory.FOOD, ExpenseType.VARIABLE),
        ("카카오T 택시", "카카오모빌리티", ExpenseCategory.TRANSPORTATION, ExpenseType.VARIABLE),
        ("KTX 승차권 예매", "코레일", ExpenseCategory.TRANSPORTATION, ExpenseType.VARIABLE),
        ("아파트 관리비 납부", "아파트아이", ExpenseCategory.HOUSING, ExpenseType.FIXED),
        ("SKT 통신요금 자동이체", "SK텔레콤", ExpenseCategory.COMMUNICATION, ExpenseType.FIXED),
        ("약국 약제비", "온누리약국", ExpenseCategory.MEDICAL, ExpenseType.VARIABLE),
        ("인프런 강의 결제", "인프런", ExpenseCategory.EDUCATION, ExpenseType.FIXED),
        ("쿠팡 로켓배송", "쿠팡", ExpenseCategory.SHOPPING, ExpenseType.VARIABLE),
        ("CGV 영화 예매", "CGV", ExpenseCategory.LEISURE, ExpenseType.VARIABLE),
    ],
)
def test_rule_category_patterns(
    classification_service, description, merchant, expected_category, expected_type
):
    """일반 소비 카테고리 정규식 키워드 분류 검증"""
    txn = create_txn(1, description=description, merchant_name=merchant)
    res = classification_service.classify_transactions([txn])[0]

    assert res.isConsumption is True
    assert res.category == expected_category
    assert res.expenseType == expected_type


# =========================================================================
# 5. 미분류 건 LLM 이관 및 1:1 보장 Fallback 테스트
# =========================================================================

def test_unclassified_transaction_transfers_to_llm(classification_service, mock_llm_service):
    """규칙에 걸리지 않는 미분류 거래는 LLM 서비스로 넘어가는지 검증"""
    txn = create_txn(99, description="알 수 없는 상점 123", merchant_name="기타상점")

    # LLM Mock 응답 설정
    mock_llm_service.classify_with_llm.return_value = [
        TransactionClassificationResult(
            transactionId=99,
            isConsumption=True,
            category=ExpenseCategory.ETC,
            expenseType=ExpenseType.VARIABLE,
        )
    ]

    results = classification_service.classify_transactions([txn])

    mock_llm_service.classify_with_llm.assert_called_once()
    assert len(results) == 1
    assert results[0].transactionId == 99


def test_1_to_1_mapping_fallback_guarantee(classification_service, mock_llm_service):
    """LLM 장애나 미응답 시에도 모든 입력 transactionId에 대한 기본 결과(Fallback) 생성 보장 검증"""
    txns = [
        create_txn(1, description="스타벅스"),  # 규칙 성공
        create_txn(2, description="알수없는 거래"),  # 규칙 실패 -> LLM 누락 발생 가정
    ]

    # LLM이 거래 2를 누락하고 빈 결과를 반환했다고 가정
    mock_llm_service.classify_with_llm.return_value = []

    results = classification_service.classify_transactions(txns)

    assert len(results) == 2
    res_map = {r.transactionId: r for r in results}

    assert 1 in res_map
    assert 2 in res_map
    # 누락된 거래 2번은 Fallback 기본값(소비/ETC/VARIABLE)으로 채워져야 함
    assert res_map[2].isConsumption is True
    assert res_map[2].category == ExpenseCategory.ETC
    assert res_map[2].expenseType == ExpenseType.VARIABLE