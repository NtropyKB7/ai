from app.rag.product_tagger import ProductTagger
from app.schemas.product_knowledge import NormalizedFinancialProduct


def test_build_tags_for_savings_njob_product():
    """
    적금/저축/N잡 관련 키워드가 있을 때
    관련 태그가 잘 생성되는지 확인합니다.
    """
    tagger = ProductTagger()

    product = NormalizedFinancialProduct(
        product_id="SAVINGS_001",
        product_name="KB N챌린지 자유적금",
        product_type="SAVINGS",
        provider="KB국민은행",
        summary="적금 금리 혜택과 저축 지원 기능 제공",
        target_group="N잡러",
        njob_trend_tip="부수입과 프리랜서 소득을 적금으로 연결해 보세요.",
        details={"interest_rate": 3.8},
        tags=[],
        is_active=True,
    )

    tags = tagger.build_tags(product)

    assert "type:savings" in tags
    assert "target:n잡러" in tags
    assert "benefit:saving" in tags
    assert "persona:njob" in tags


def test_build_tags_for_food_and_transport_card():
    """
    식비/교통 관련 키워드가 있을 때
    소비 카테고리 태그가 생성되는지 확인합니다.
    """
    tagger = ProductTagger()

    product = NormalizedFinancialProduct(
        product_id="CARD_001",
        product_name="생활 교통 할인 카드",
        product_type="CARD",
        provider="KB국민카드",
        summary="식비, 배달, 교통, 대중교통 할인 제공",
        target_group="생활비 관리형 사용자",
        njob_trend_tip="",
        details={"discount_rate": 0.1},
        tags=[],
        is_active=True,
    )

    tags = tagger.build_tags(product)

    assert "type:card" in tags
    assert "expense:food" in tags
    assert "expense:transportation" in tags


def test_build_tags_for_long_term_product():
    """
    장기/만기/목돈 관련 키워드가 있으면 장기 목적 태그가 생성되는지 확인합니다.
    """
    tagger = ProductTagger()

    product = NormalizedFinancialProduct(
        product_id="SAVINGS_002",
        product_name="목돈 마련 장기 적금",
        product_type="SAVINGS",
        provider="테스트은행",
        summary="만기까지 유지 시 장기 저축에 유리한 상품",
        target_group="사회초년생",
        njob_trend_tip="장기 부수입 관리에 적합합니다.",
        details={},
        tags=[],
        is_active=True,
    )

    tags = tagger.build_tags(product)

    assert "term:long" in tags
    assert "benefit:saving" in tags