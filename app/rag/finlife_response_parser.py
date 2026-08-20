from __future__ import annotations

from app.schemas.finlife_product import (
    FinlifeDepositResponse,
    FinlifePage,
    FinlifeSavingsResponse,
    ProductType,
)


class FinlifeResponseError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"Finlife response failed: {code} {message}")
        self.code = code
        self.message = message


def parse_finlife_response(payload: dict, expected_type: ProductType) -> FinlifePage:
    if expected_type == ProductType.SAVINGS:
        result = FinlifeSavingsResponse.model_validate(payload).result
        expected_div = "S"
    elif expected_type == ProductType.DEPOSIT:
        result = FinlifeDepositResponse.model_validate(payload).result
        expected_div = "D"
    else:
        raise ValueError("Finlife response supports SAVINGS or DEPOSIT only")

    if not result.is_success:
        raise FinlifeResponseError(result.err_cd, result.err_msg)
    if result.prdt_div is not None and result.prdt_div != expected_div:
        raise ValueError("prdt_div and requested product type do not match")
    required = (result.total_count, result.max_page_no, result.now_page_no)
    if any(value is None for value in required):
        raise ValueError("Successful response is missing pagination fields")

    return FinlifePage(
        product_type=expected_type,
        total_count=result.total_count,
        max_page_no=result.max_page_no,
        now_page_no=result.now_page_no,
        bases=result.baseList or [],
        options=result.optionList or [],
    )
