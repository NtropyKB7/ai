from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ExpenseCategory(str, Enum):
    """Spring Enum 및 TXN_ANALYSIS.category와 매핑되는 소비 카테고리."""

    FOOD = "FOOD"
    TRANSPORTATION = "TRANSPORTATION"
    HOUSING = "HOUSING"
    COMMUNICATION = "COMMUNICATION"
    MEDICAL = "MEDICAL"
    EDUCATION = "EDUCATION"
    SHOPPING = "SHOPPING"
    LEISURE = "LEISURE"
    INSURANCE = "INSURANCE"
    FINANCE = "FINANCE"
    ETC = "ETC"


class ExpenseType(str, Enum):
    """소비 지출의 반복성 유형."""

    FIXED = "FIXED"
    VARIABLE = "VARIABLE"


class TransactionCategory(str, Enum):
    """ACCOUNT_TRANSACTION.transaction_category 표준 코드."""

    ORDINARY = "ORDINARY"
    INSTALLMENT = "INSTALLMENT"
    LOAN = "LOAN"


class TransactionForClassification(BaseModel):
    """Spring AI-service가 FastAPI에 전달하는 거래 한 건."""

    transactionId: int = Field(
        ...,
        gt=0,
        description="ACCOUNT_TRANSACTION의 거래 ID",
    )

    amount: int = Field(
        ...,
        gt=0,
        description="분류 대상 거래 금액(원 단위)",
    )

    transactionCategory: TransactionCategory = Field(
        ...,
        description="ACCOUNT_TRANSACTION 거래 유형",
    )

    organizationCode: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="은행별 거래 설명 필드 해석에 사용하는 CODEF 기관코드",
    )

    desc1: Optional[str] = Field(
        default=None,
        max_length=255,
        description="CODEF resAccountDesc1 원문",
    )

    desc2: Optional[str] = Field(
        default=None,
        max_length=255,
        description="CODEF resAccountDesc2 원문",
    )

    desc3: Optional[str] = Field(
        default=None,
        max_length=255,
        description="CODEF resAccountDesc3 원문. 주로 상대방·가맹점·상품명",
    )

    desc4: Optional[str] = Field(
        default=None,
        max_length=255,
        description="CODEF resAccountDesc4 원문",
    )


class TransactionClassificationRequest(BaseModel):
    """여러 거래를 한 번에 분류하는 요청 DTO."""

    transactions: list[TransactionForClassification] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="분류할 거래 목록",
    )

    @model_validator(mode="after")
    def validate_unique_transaction_ids(self):
        transaction_ids = [
            transaction.transactionId
            for transaction in self.transactions
        ]

        if len(transaction_ids) != len(set(transaction_ids)):
            raise ValueError(
                "transactionId must be unique within a request"
            )

        return self


class TransactionClassificationResult(BaseModel):
    """거래 한 건의 소비 분류 결과."""

    transactionId: int = Field(
        ...,
        description="분류된 원본 거래 ID",
    )

    isConsumption: bool = Field(
        ...,
        description="소비 거래 여부",
    )

    category: Optional[ExpenseCategory] = Field(
        default=None,
        description="소비 카테고리. 비소비 거래는 null",
    )

    expenseType: Optional[ExpenseType] = Field(
        default=None,
        description="지출 유형. 비소비 거래는 null",
    )


class TransactionClassificationData(BaseModel):
    """공통 응답의 data 필드."""

    results: list[TransactionClassificationResult] = Field(
        ...,
        description="거래별 분류 결과",
    )


class TransactionClassificationResponse(BaseModel):
    """소비 거래 일괄 분류 API 응답 DTO."""

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
        description="소비 거래 분류 결과 데이터",
    )


class LLMTransactionClassificationResponse(BaseModel):
    """LLM 보조 분류 응답 형식."""

    results: list[TransactionClassificationResult] = Field(
        ...,
        description="LLM 보조 분류 결과",
    )