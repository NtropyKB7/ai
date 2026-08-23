from dataclasses import dataclass

from app.schemas.transaction import TransactionForClassification


@dataclass(frozen=True)
class TransactionDescriptionContext:
    """원본 거래 설명을 변경하지 않고 분류에 사용할 문맥."""

    original_description: str | None
    payment_method: str | None
    merchant_candidate: str | None


def normalize_transaction_description(
    transaction: TransactionForClassification,
) -> TransactionDescriptionContext:
    """
    원천 계약의 ``결제수단_가맹점명`` 형식을 안전하게 분리합니다.

    언더바가 정확히 하나이고 양쪽 값이 모두 있을 때만 계약에 맞는
    값으로 인정합니다. 그 외 형식은 추측하지 않고 desc3 원문을
    가맹점 후보로 유지합니다.
    """
    original = transaction.desc3
    normalized = original.strip() if original and original.strip() else None

    if normalized and normalized.count("_") == 1:
        payment_method, merchant_candidate = (
            value.strip() for value in normalized.split("_", maxsplit=1)
        )
        if payment_method and merchant_candidate:
            return TransactionDescriptionContext(
                original_description=original,
                payment_method=payment_method,
                merchant_candidate=merchant_candidate,
            )

    return TransactionDescriptionContext(
        original_description=original,
        payment_method=None,
        merchant_candidate=normalized,
    )
