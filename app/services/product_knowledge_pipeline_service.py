from app.core.config import settings
from app.rag.chroma_client import chroma_manager
from app.rag.product_document_builder import ProductDocumentBuilder
from app.rag.product_normalizer import ProductNormalizer
from app.rag.product_source_loader import ProductSourceLoader
from app.rag.product_tagger import ProductTagger
from app.schemas.product_knowledge import NormalizedFinancialProduct


class ProductKnowledgePipelineService:
    """
    금융상품 지식 DB 갱신 파이프라인 서비스입니다.

    처리 흐름:
    1. 원천 데이터 로드
    2. 정제/정규화
    3. 태그 생성
    4. 임베딩용 문서 생성
    5. ChromaDB upsert
    """

    def __init__(self):
        self.source_loader = ProductSourceLoader()
        self.normalizer = ProductNormalizer()
        self.tagger = ProductTagger()
        self.document_builder = ProductDocumentBuilder()

    def refresh_from_seed_file(self) -> dict:
        """
        seed_products.json 기반으로 금융상품 지식 DB를 갱신합니다.

        Returns:
            파이프라인 실행 결과 요약
        """
        raw_products = self.source_loader.load_from_json_file(
            settings.FINANCIAL_PRODUCT_SOURCE_FILE
        )

        normalized_products: list[NormalizedFinancialProduct] = []

        for raw_product in raw_products:
            normalized = self.normalizer.normalize(raw_product)
            normalized.tags = self.tagger.build_tags(normalized)
            normalized_products.append(normalized)

        documents = [
            self.document_builder.build(product)
            for product in normalized_products
            if product.is_active
        ]

        upsert_count = chroma_manager.upsert_product_documents(documents)

        return {
            "source_count": len(raw_products),
            "normalized_count": len(normalized_products),
            "upserted_count": upsert_count,
        }


product_knowledge_pipeline_service = ProductKnowledgePipelineService()