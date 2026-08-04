# 날짜·시간 문자열을 FastAPI가 검증된 datetime 객체로 변환하기 위해 사용합니다.
from datetime import datetime

# Enum은 정해진 값만 입력·반환되도록 제한하는 타입입니다.
from enum import Enum

# Optional은 값이 없을 때 None(null)을 허용하기 위해 사용합니다.
from typing import Optional

# BaseModel은 FastAPI 요청·응답 데이터의 형식을 정의합니다.
# Field는 값 검증과 Swagger 문서 설명에 사용합니다.
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    """
    거래의 입출금 유형입니다.

    str을 함께 상속하면 JSON 응답에서
    "TransactionType.IN"이 아니라 "IN" 문자열로 반환됩니다.
    """

    # 입금 거래
    IN = "IN"

    # 출금 거래
    OUT = "OUT"


class ExpenseCategory(str, Enum):
    """
    소비 거래의 지출 카테고리입니다.

    이 값은 Spring account-service가 TXN_ANALYSIS.category에
    저장할 표준 코드입니다.
    """

    FOOD = "FOOD"                    # 식비
    TRANSPORT = "TRANSPORT"          # 교통·주유
    HOUSING = "HOUSING"              # 주거·통신
    HEALTH = "HEALTH"                # 의료·건강
    SHOPPING = "SHOPPING"            # 쇼핑
    LEISURE = "LEISURE"              # 취미·여가
    SUBSCRIPTION = "SUBSCRIPTION"    # OTT 등 정기구독
    EDUCATION = "EDUCATION"          # 교육
    FINANCE = "FINANCE"              # 보험료·수수료 등 금융성 지출
    OTHER = "OTHER"                  # 규칙으로 분류하기 어려운 소비


class ExpenseType(str, Enum):
    """
    소비 지출의 반복성 유형입니다.

    FIXED: 매월 또는 일정 주기로 반복되는 고정성 지출
    VARIABLE: 소비 패턴에 따라 금액과 횟수가 달라지는 지출
    """

    FIXED = "FIXED"
    VARIABLE = "VARIABLE"


class TransactionForClassification(BaseModel):
    """
    Spring AI-service가 FastAPI에 전달하는 거래 내역 1건의 형식입니다.

    실제 거래 데이터의 소유자는 Spring account-service이며,
    FastAPI는 이 데이터를 받아 분류만 수행합니다.
    """

    # ACCOUNT_TRANSACTION의 거래 ID입니다.
    # FastAPI가 분류 결과를 반환할 때 같은 ID를 함께 반환합니다.
    transactionId: int = Field(
        ...,
        description="ACCOUNT_TRANSACTION의 거래 ID",
    )

    # ISO 8601 형식의 문자열을 받습니다.
    # 예: "2026-08-01T12:30:00"
    # FastAPI/Pydantic이 datetime으로 자동 변환합니다.
    transactionDate: datetime = Field(
        ...,
        description="거래 일시",
    )

    # 입금(IN) 또는 출금(OUT) 여부입니다.
    txnType: TransactionType = Field(
        ...,
        description="입출금 구분: IN 또는 OUT",
    )

    # 거래 금액입니다.
    # ge=0은 음수 금액을 요청으로 받지 않도록 검증합니다.
    amount: int = Field(
        ...,
        ge=0,
        description="거래 금액(원 단위)",
    )

    # CODEF 거래처명입니다.
    # 이자 입금 등에서는 빈 문자열 또는 null일 수 있습니다.
    merchantName: Optional[str] = Field(
        default=None,
        description="거래처명, 없으면 null 또는 빈 문자열",
    )

    # CODEF 거래 내용입니다.
    # merchantName이 비어 있는 경우 중요한 분류 보조 정보가 됩니다.
    transactionDetails: Optional[str] = Field(
        default=None,
        description="거래 내용, 없으면 null 또는 빈 문자열",
    )

    # 거래와 연결된 N잡 ID입니다.
    # 연결되지 않은 거래는 null입니다.
    jobId: Optional[int] = Field(
        default=None,
        description="매칭된 잡 ID, 없으면 null",
    )

    # 거래와 연결된 N잡 이름입니다.
    jobName: Optional[str] = Field(
        default=None,
        description="매칭된 잡 이름, 없으면 null",
    )


class TransactionClassificationRequest(BaseModel):
    """
    여러 거래 내역을 한 번에 분류하기 위한 FastAPI 요청 DTO입니다.

    거래마다 API를 호출하지 않고 bulk 요청을 사용해
    네트워크 호출 횟수를 줄입니다.
    """

    transactions: list[TransactionForClassification] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="분류할 거래 내역 목록",
    )


class TransactionClassificationResult(BaseModel):
    """
    거래 내역 1건에 대한 FastAPI 분류 결과 DTO입니다.

    이 객체는 FastAPI가 Spring으로 반환하며,
    Spring account-service가 TXN_ANALYSIS 저장에 사용합니다.
    """

    # 원본 거래와 분류 결과를 연결하기 위한 거래 ID입니다.
    transactionId: int = Field(
        ...,
        description="분류된 원본 거래 ID",
    )

    # 소비 거래 여부입니다.
    isConsumption: bool = Field(
        ...,
        description="소비 거래 여부",
    )

    # 비소비 거래라면 null을 반환합니다.
    category: Optional[ExpenseCategory] = Field(
        default=None,
        description="소비 카테고리, 비소비 거래면 null",
    )

    # 비소비 거래라면 null을 반환합니다.
    expenseType: Optional[ExpenseType] = Field(
        default=None,
        description="FIXED 또는 VARIABLE, 비소비 거래면 null",
    )

    # 분류 확신도입니다.
    # 0.0부터 1.0까지만 허용합니다.
    # 이후 OTHER 또는 낮은 confidence 거래를 LLM으로 보조 분류할 때 사용합니다.
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="분류 신뢰도, 0.0 ~ 1.0",
    )


class TransactionClassificationResponse(BaseModel):
    """
    소비 내역 일괄 분류 API의 최종 응답 DTO입니다.
    """

    results: list[TransactionClassificationResult] = Field(
        ...,
        description="거래별 분류 결과 목록",
    )