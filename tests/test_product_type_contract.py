import pytest
from pydantic import ValidationError

from app.schemas.product_knowledge import RawFinancialProduct


@pytest.mark.parametrize("value", ["CARD", "SAVINGS", "DEPOSIT", " savings "])
def test_supported_product_types(value):
    item = RawFinancialProduct(
        product_id="id",
        product_name="name",
        product_type=value,
        provider="provider",
    )
    assert item.product_type in {"CARD", "SAVINGS", "DEPOSIT"}


def test_rejects_unknown_product_type():
    with pytest.raises(ValidationError):
        RawFinancialProduct(
            product_id="id",
            product_name="name",
            product_type="LOAN",
            provider="provider",
        )
