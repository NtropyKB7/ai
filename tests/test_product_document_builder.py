from app.rag.product_document_builder import ProductDocumentBuilder
from app.schemas.product_knowledge import NormalizedFinancialProduct


def test_build_document_contains_core_fields():
    """
    임베딩용 document_text에 핵심 필드가 잘 포함되는지 확인합니다.
    """
    builder = ProductDocumentBuilder()

    product = NormalizedFinancialProduct(
        product_id="SAVINGS_001",
        product_name="KB N챌린지 자유적금",
        product_type="SAVINGS",
        provider="KB국민은행",
        summary="자동이체 조건 충족 시 우대금리를 제공하는 자유적금 상품",
        target_group="N잡러",
        njob_trend_tip="부수입 일부를 적금으로 연결해 보세요.",
        details={
            "interest_rate": 3.8,
            "saving_period": 12,
            "max_monthly_amount": 500000,
        },
        tags=["type:savings", "benefit:saving", "persona:njob"],
        is_active=True,
    )

    document = builder.build(product)

    assert document.product_id == "SAVINGS_001"
    assert "상품명: KB N챌린지 자유적금" in document.document_text
    assert "상품유형: SAVINGS" in document.document_text
    assert "금융사: KB국민은행" in document.document_text
    assert "핵심혜택: 자동이체 조건 충족 시 우대금리를 제공하는 자유적금 상품" in document.document_text
    assert "추천대상: N잡러" in document.document_text
    assert "N잡활용팁: 부수입 일부를 적금으로 연결해 보세요." in document.document_text
    assert "태그: type:savings, benefit:saving, persona:njob" in document.document_text


def test_build_document_metadata_contains_expected_fields():
    """
    metadata에 검색/복원에 필요한 필드가 잘 들어가는지 확인합니다.
    """
    builder = ProductDocumentBuilder()

    product = NormalizedFinancialProduct(
        product_id="CARD_001",
        product_name="생활혜택 카드",
        product_type="CARD",
        provider="KB국민카드",
        summary="생활비 할인 혜택 제공",
        target_group="생활비 관리형 사용자",
        njob_trend_tip="",
        details={"discount_rate": 0.1, "annual_fee": 15000},
        tags=["type:card", "expense:food"],
        is_active=True,
    )

    document = builder.build(product)

    metadata = document.metadata

    assert metadata["product_id"] == "CARD_001"
    assert metadata["product_name"] == "생활혜택 카드"
    assert metadata["product_type"] == "CARD"
    assert metadata["provider"] == "KB국민카드"
    assert metadata["summary"] == "생활비 할인 혜택 제공"
    assert metadata["target_group"] == "생활비 관리형 사용자"
    assert metadata["tags"] == "type:card,expense:food"
    assert metadata["is_active"] == "true"
    assert "details_json" in metadata


def test_extract_details_text_uses_allowed_keys_only():
    """
    details에서 허용된 키만 임베딩용 상세정보 텍스트로 들어가는지 확인합니다.
    """
    builder = ProductDocumentBuilder()

    product = NormalizedFinancialProduct(
        product_id="CARD_002",
        product_name="테스트 카드",
        product_type="CARD",
        provider="테스트카드사",
        summary="테스트 요약",
        target_group="",
        njob_trend_tip="",
        details={
            "annual_fee": 10000,
            "discount_rate": 0.15,
            "unknown_key": "should_not_be_included",
        },
        tags=[],
        is_active=True,
    )

    document = builder.build(product)

    assert "annual_fee=10000" in document.document_text
    assert "discount_rate=0.15" in document.document_text
    assert "unknown_key" not in document.document_text