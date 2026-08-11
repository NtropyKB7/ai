from app.rag.product_normalizer import ProductNormalizer
from app.schemas.product_knowledge import RawFinancialProduct


def test_normalize_trims_and_uppercases_product_type():
    """
    공백 제거와 상품 유형 대문자 정규화가 잘 되는지 확인합니다.
    """
    normalizer = ProductNormalizer()

    raw_product = RawFinancialProduct(
        product_id="  SAVINGS_001  ",
        product_name="  KB N챌린지 자유적금  ",
        product_type="  savings  ",
        provider="  KB국민은행  ",
        summary="  자동이체 조건 충족 시 우대금리를 제공하는 자유적금 상품  ",
        target_group="  N잡러  ",
        njob_trend_tip="  부수입 일부를 자동이체해 보세요.  ",
        details={"interest_rate": 3.8},
    )

    normalized = normalizer.normalize(raw_product)

    assert normalized.product_id == "SAVINGS_001"
    assert normalized.product_name == "KB N챌린지 자유적금"
    assert normalized.product_type == "SAVINGS"
    assert normalized.provider == "KB국민은행"
    assert normalized.summary == "자동이체 조건 충족 시 우대금리를 제공하는 자유적금 상품"
    assert normalized.target_group == "N잡러"
    assert normalized.njob_trend_tip == "부수입 일부를 자동이체해 보세요."
    assert normalized.details == {"interest_rate": 3.8}


def test_normalize_fills_optional_fields_with_empty_string():
    """
    optional 문자열 필드가 None이어도 빈 문자열로 보정되는지 확인합니다.
    """
    normalizer = ProductNormalizer()

    raw_product = RawFinancialProduct(
        product_id="CARD_001",
        product_name="KB 생활혜택 카드",
        product_type="card",
        provider="KB국민카드",
        summary="생활비 할인 혜택 제공",
        target_group=None,
        njob_trend_tip=None,
        details={},
    )

    normalized = normalizer.normalize(raw_product)

    assert normalized.target_group == ""
    assert normalized.njob_trend_tip == ""
    assert normalized.tags == []
    assert normalized.is_active is True


def test_normalize_handles_empty_details():
    """
    details가 비어 있어도 빈 dict로 안전하게 유지되는지 확인합니다.
    """
    normalizer = ProductNormalizer()

    raw_product = RawFinancialProduct(
        product_id="CARD_002",
        product_name="테스트 카드",
        product_type="CARD",
        provider="테스트카드사",
        summary="테스트 혜택",
        details={},
    )

    normalized = normalizer.normalize(raw_product)

    assert normalized.details == {}