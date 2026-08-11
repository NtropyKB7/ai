from app.schemas.product_knowledge import (
    NormalizedFinancialProduct,
    ProductKnowledgeDocument,
    RawFinancialProduct,
)
from app.services.product_knowledge_pipeline_service import (
    ProductKnowledgePipelineService,
)


def test_refresh_from_seed_file_runs_full_pipeline(monkeypatch):
    """
    파이프라인 서비스가
    1) 원천 데이터 로드
    2) 정규화
    3) 태그 생성
    4) 문서 생성
    5) ChromaDB upsert
    흐름으로 정상 동작하는지 확인합니다.
    """

    service = ProductKnowledgePipelineService()

    raw_products = [
        RawFinancialProduct(
            product_id="SAVINGS_001",
            product_name="KB N챌린지 자유적금",
            product_type="savings",
            provider="KB국민은행",
            summary="적금 금리 혜택 제공",
            target_group="N잡러",
            njob_trend_tip="부수입 일부를 적금으로 연결",
            details={"interest_rate": 3.8},
        )
    ]

    normalized_product = NormalizedFinancialProduct(
        product_id="SAVINGS_001",
        product_name="KB N챌린지 자유적금",
        product_type="SAVINGS",
        provider="KB국민은행",
        summary="적금 금리 혜택 제공",
        target_group="N잡러",
        njob_trend_tip="부수입 일부를 적금으로 연결",
        details={"interest_rate": 3.8},
        tags=["type:savings", "benefit:saving", "persona:njob"],
        is_active=True,
    )

    built_document = ProductKnowledgeDocument(
        product_id="SAVINGS_001",
        document_text="상품명: KB N챌린지 자유적금 | 상품유형: SAVINGS",
        metadata={
            "product_id": "SAVINGS_001",
            "product_name": "KB N챌린지 자유적금",
            "product_type": "SAVINGS",
            "provider": "KB국민은행",
            "summary": "적금 금리 혜택 제공",
            "target_group": "N잡러",
            "njob_trend_tip": "부수입 일부를 적금으로 연결",
            "tags": "type:savings,benefit:saving,persona:njob",
            "is_active": "true",
            "details_json": "{\"interest_rate\": 3.8}",
        },
    )

    def fake_load_from_json_file(_path: str):
        return raw_products

    def fake_normalize(raw_product: RawFinancialProduct):
        assert raw_product.product_id == "SAVINGS_001"
        return normalized_product

    def fake_build_tags(product: NormalizedFinancialProduct):
        assert product.product_id == "SAVINGS_001"
        return ["type:savings", "benefit:saving", "persona:njob"]

    def fake_build(product: NormalizedFinancialProduct):
        assert product.tags == ["type:savings", "benefit:saving", "persona:njob"]
        return built_document

    def fake_upsert_product_documents(documents: list[ProductKnowledgeDocument]):
        assert len(documents) == 1
        assert documents[0].product_id == "SAVINGS_001"
        return 1

    monkeypatch.setattr(
        service.source_loader,
        "load_from_json_file",
        fake_load_from_json_file,
    )
    monkeypatch.setattr(
        service.normalizer,
        "normalize",
        fake_normalize,
    )
    monkeypatch.setattr(
        service.tagger,
        "build_tags",
        fake_build_tags,
    )
    monkeypatch.setattr(
        service.document_builder,
        "build",
        fake_build,
    )

    # 모듈 전역 chroma_manager를 service 파일에서 직접 import해서 쓰는 구조를 가정
    from app.services import product_knowledge_pipeline_service as pipeline_module

    monkeypatch.setattr(
        pipeline_module.chroma_manager,
        "upsert_product_documents",
        fake_upsert_product_documents,
    )

    monkeypatch.setattr(
        pipeline_module.settings,
        "FINANCIAL_PRODUCT_SOURCE_FILE",
        "app/rag/seed_products.json",
    )

    result = service.refresh_from_seed_file()

    assert result["source_count"] == 1
    assert result["normalized_count"] == 1
    assert result["upserted_count"] == 1


def test_refresh_from_seed_file_skips_inactive_products(monkeypatch):
    """
    비활성 상품은 문서 생성 후 upsert 대상에서 제외되는지 확인합니다.
    """

    service = ProductKnowledgePipelineService()

    raw_products = [
        RawFinancialProduct(
            product_id="CARD_999",
            product_name="비활성 상품",
            product_type="card",
            provider="테스트카드사",
            summary="테스트 상품",
            details={},
        )
    ]

    inactive_product = NormalizedFinancialProduct(
        product_id="CARD_999",
        product_name="비활성 상품",
        product_type="CARD",
        provider="테스트카드사",
        summary="테스트 상품",
        target_group="",
        njob_trend_tip="",
        details={},
        tags=["type:card"],
        is_active=False,
    )

    def fake_load_from_json_file(_path: str):
        return raw_products

    def fake_normalize(_raw_product: RawFinancialProduct):
        return inactive_product

    def fake_build_tags(_product: NormalizedFinancialProduct):
        return ["type:card"]

    def fake_upsert_product_documents(documents: list[ProductKnowledgeDocument]):
        # 비활성 상품이므로 upsert 대상 문서가 없어야 함
        assert documents == []
        return 0

    monkeypatch.setattr(
        service.source_loader,
        "load_from_json_file",
        fake_load_from_json_file,
    )
    monkeypatch.setattr(
        service.normalizer,
        "normalize",
        fake_normalize,
    )
    monkeypatch.setattr(
        service.tagger,
        "build_tags",
        fake_build_tags,
    )

    from app.services import product_knowledge_pipeline_service as pipeline_module

    monkeypatch.setattr(
        pipeline_module.chroma_manager,
        "upsert_product_documents",
        fake_upsert_product_documents,
    )

    monkeypatch.setattr(
        pipeline_module.settings,
        "FINANCIAL_PRODUCT_SOURCE_FILE",
        "app/rag/seed_products.json",
    )

    result = service.refresh_from_seed_file()

    assert result["source_count"] == 1
    assert result["normalized_count"] == 1
    assert result["upserted_count"] == 0


def test_refresh_from_seed_file_propagates_multiple_products(monkeypatch):
    """
    여러 상품이 들어와도 normalized/upsert 카운트가 기대대로 집계되는지 확인합니다.
    """

    service = ProductKnowledgePipelineService()

    raw_products = [
        RawFinancialProduct(
            product_id="SAVINGS_001",
            product_name="상품1",
            product_type="savings",
            provider="은행1",
            summary="적금 상품",
            details={},
        ),
        RawFinancialProduct(
            product_id="CARD_001",
            product_name="상품2",
            product_type="card",
            provider="카드사1",
            summary="교통 할인 카드",
            details={},
        ),
    ]

    normalized_products = [
        NormalizedFinancialProduct(
            product_id="SAVINGS_001",
            product_name="상품1",
            product_type="SAVINGS",
            provider="은행1",
            summary="적금 상품",
            target_group="",
            njob_trend_tip="",
            details={},
            tags=[],
            is_active=True,
        ),
        NormalizedFinancialProduct(
            product_id="CARD_001",
            product_name="상품2",
            product_type="CARD",
            provider="카드사1",
            summary="교통 할인 카드",
            target_group="",
            njob_trend_tip="",
            details={},
            tags=[],
            is_active=True,
        ),
    ]

    call_index = {"value": 0}

    def fake_load_from_json_file(_path: str):
        return raw_products

    def fake_normalize(_raw_product: RawFinancialProduct):
        product = normalized_products[call_index["value"]]
        call_index["value"] += 1
        return product

    def fake_build_tags(product: NormalizedFinancialProduct):
        return [f"type:{product.product_type.lower()}"]

    def fake_build(product: NormalizedFinancialProduct):
        return ProductKnowledgeDocument(
            product_id=product.product_id,
            document_text=f"상품명: {product.product_name}",
            metadata={"product_id": product.product_id},
        )

    def fake_upsert_product_documents(documents: list[ProductKnowledgeDocument]):
        assert len(documents) == 2
        return 2

    monkeypatch.setattr(
        service.source_loader,
        "load_from_json_file",
        fake_load_from_json_file,
    )
    monkeypatch.setattr(
        service.normalizer,
        "normalize",
        fake_normalize,
    )
    monkeypatch.setattr(
        service.tagger,
        "build_tags",
        fake_build_tags,
    )
    monkeypatch.setattr(
        service.document_builder,
        "build",
        fake_build,
    )

    from app.services import product_knowledge_pipeline_service as pipeline_module

    monkeypatch.setattr(
        pipeline_module.chroma_manager,
        "upsert_product_documents",
        fake_upsert_product_documents,
    )

    monkeypatch.setattr(
        pipeline_module.settings,
        "FINANCIAL_PRODUCT_SOURCE_FILE",
        "app/rag/seed_products.json",
    )

    result = service.refresh_from_seed_file()

    assert result["source_count"] == 2
    assert result["normalized_count"] == 2
    assert result["upserted_count"] == 2