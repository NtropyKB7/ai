from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ==========================================
# Enums
# ==========================================

class ExpenseCategory(str, Enum):
    """
    소비 거래의 지출 카테고리입니다.

    Spring Java Enum 및 TXN_ANALYSIS.category와 1:1 매핑되는 표준 코드입니다.
    """
    FOOD = "FOOD"                    # 식비
    TRANSPORTATION = "TRANSPORTATION"# 교통·주유
    HOUSING = "HOUSING"              # 주거
    COMMUNICATION = "COMMUNICATION"  # 통신
    MEDICAL = "MEDICAL"              # 의료·건강
    EDUCATION = "EDUCATION"          # 교육
    SHOPPING = "SHOPPING"            # 쇼핑
    LEISURE = "LEISURE"              # 취미·여가
    INSURANCE = "INSURANCE"          # 보험
    FINANCE = "FINANCE"              # 대출 이자 등 금융 지출
    ETC = "ETC"                      # 카드대금 등 기타 소비


class ExpenseType(str, Enum):
    """
    소비 지출의 반복성 유형입니다.

    FIXED: 매월 또는 일정 주기로 반복되는 고정성 지출
    VARIABLE: 소비 패턴에 따라 금액과 횟수가 달라지는 지출
    """
    FIXED = "FIXED"
    VARIABLE = "VARIABLE"


# ==========================================
# Request DTOs
# ==========================================

class TransactionForClassification(BaseModel):
    """
    Spring AI-service가 FastAPI에 전달하는 거래 내역 1건의 형식입니다.
    (출금 거래만 전달됩니다.)
    """

    transactionId: int = Field(
        ...,
        description="ACCOUNT_TRANSACTION의 거래 ID",
    )

    transactionDate: datetime = Field(
        ...,
        description="거래 일시 (ISO 8601 문자열)",
    )

    amount: int = Field(
        ...,
        ge=0,
        description="거래 금액(원 단위)",
    )

    merchantName: Optional[str] = Field(
        default=None,
        description="거래처명 / 가맹점명, 없으면 null 또는 빈 문자열",
    )

    description: str = Field(
        ...,
        description="거래 상세 설명 / 적요 조합 문자열",
    )


class TransactionClassificationRequest(BaseModel):
    """
    여러 거래 내역을 한 번에 분류하기 위한 FastAPI 요청 DTO입니다.
    """

    transactions: list[TransactionForClassification] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="분류할 거래 내역 목록",
    )


# ==========================================
# Response DTOs
# ==========================================

class TransactionClassificationResult(BaseModel):
    """
    거래 내역 1건에 대한 FastAPI 분류 결과 DTO입니다.
    """

    transactionId: int = Field(
        ...,
        description="분류된 원본 거래 ID",
    )

    isConsumption: bool = Field(
        ...,
        description="소비 거래 여부 (True: 소비, False: 비소비)",
    )

    category: Optional[ExpenseCategory] = Field(
        default=None,
        description="소비 카테고리, 비소비 거래면 null",
    )

    expenseType: Optional[ExpenseType] = Field(
        default=None,
        description="FIXED 또는 VARIABLE, 비소비 거래면 null",
    )


class TransactionClassificationData(BaseModel):
    """
    공통 API 응답의 data 내부에 들어갈 실제 분류 결과 목록입니다.
    """

    results: list[TransactionClassificationResult] = Field(
        ...,
        description="거래별 분류 결과 목록",
    )


class TransactionClassificationResponse(BaseModel):
    """
    소비 내역 일괄 분류 API의 최종 성공 응답 DTO입니다. (팀 공통 응답 형식)
    """

    success: bool = Field(
        ...,
        description="요청 성공 여부",
    )

    status_code: int = Field(
        ...,
        description="HTTP 상태 코드",
    )

    message: str = Field(
        ...,
        description="응답 메시지",
    )

    data: TransactionClassificationData = Field(
        ...,
        description="소비 내역 분류 결과 데이터",
    )


class LLMTransactionClassificationResponse(BaseModel):
    """
    LLM이 보조 분류 시 내부적으로 반환받는 JSON 형식입니다.
    """

    results: list[TransactionClassificationResult] = Field(
        ...,
        description="LLM이 보조 분류한 거래 결과 목록",
    )