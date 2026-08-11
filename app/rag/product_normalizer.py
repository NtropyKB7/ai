from app.schemas.product_knowledge import (
    NormalizedFinancialProduct,
    RawFinancialProduct,
)


class ProductNormalizer:
    """
    금융상품 원천 데이터를 정제/정규화하는 클래스입니다.

    목적:
    - 누락값을 기본값으로 보정
    - 문자열 앞뒤 공백 제거
    - summary, provider, target_group 등을 검색/추천에 적합하게 정리
    - product_type 값을 일관된 형식으로 맞춤
    """

    def normalize(
        self,
        raw_product: RawFinancialProduct,
    ) -> NormalizedFinancialProduct:
        """
        원천 상품 DTO를 정규화된 상품 DTO로 변환합니다.
        """
        product_type = (raw_product.product_type or "").strip().upper()

        return NormalizedFinancialProduct(
            product_id=raw_product.product_id.strip(),
            product_name=(raw_product.product_name or "").strip(),
            product_type=product_type,
            provider=(raw_product.provider or "").strip(),
            summary=(raw_product.summary or "").strip(),
            target_group=(raw_product.target_group or "").strip(),
            njob_trend_tip=(raw_product.njob_trend_tip or "").strip(),
            details=raw_product.details or {},
            tags=[],
            is_active=True,
        )