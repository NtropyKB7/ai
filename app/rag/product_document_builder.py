import json

from app.schemas.product_knowledge import (
    NormalizedFinancialProduct,
    ProductKnowledgeDocument,
)


class ProductDocumentBuilder:
    """
    금융상품 임베딩용 문서를 생성하는 클래스입니다.

    핵심 원칙:
    - 임베딩에 필요한 정보만 본문에 넣습니다.
    - 구조적으로 필터링 가능한 정보는 metadata에도 함께 넣습니다.
    - 단순 전체 문자열 덤프보다 추천/검색에 유리한 문장 구조를 만듭니다.
    """

    def build(
        self,
        product: NormalizedFinancialProduct,
    ) -> ProductKnowledgeDocument:
        """
        정규화된 금융상품을 ChromaDB upsert용 문서로 변환합니다.
        """
        document_parts = [
            f"상품명: {product.product_name}",
            f"상품유형: {product.product_type}",
            f"금융사: {product.provider}",
        ]

        if product.summary:
            document_parts.append(f"핵심혜택: {product.summary}")

        if product.target_group:
            document_parts.append(f"추천대상: {product.target_group}")

        if product.njob_trend_tip:
            document_parts.append(f"N잡활용팁: {product.njob_trend_tip}")

        if product.tags:
            document_parts.append(f"태그: {', '.join(product.tags)}")

        # details는 전부 임베딩 본문에 넣기보다
        # 검색에 도움이 될 핵심 정보만 텍스트화하고,
        # 전체는 metadata/details_json으로 보관합니다.
        details_text = self._extract_details_text(product.details)
        if details_text:
            document_parts.append(f"상세정보: {details_text}")

        document_text = " | ".join(document_parts)

        metadata = {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "product_type": product.product_type,
            "provider": product.provider,
            "summary": product.summary,
            "target_group": product.target_group,
            "njob_trend_tip": product.njob_trend_tip,
            "tags": ",".join(product.tags),
            "is_active": str(product.is_active).lower(),
            "details_json": json.dumps(product.details, ensure_ascii=False),
        }

        return ProductKnowledgeDocument(
            product_id=product.product_id,
            document_text=document_text,
            metadata=metadata,
        )

    def _extract_details_text(
        self,
        details: dict,
    ) -> str:
        """
        details 딕셔너리에서 임베딩에 유용한 핵심 항목만 추출합니다.
        """
        if not details:
            return ""

        if details.get("source") == "FSS_FINLIFE":
            parts = []
            source_fields = [
                "join_way",
                "maturity_interest_text",
                "special_conditions_text",
                "join_restriction_code",
                "eligible_members_text",
                "notes_text",
                "max_limit",
            ]
            for key in source_fields:
                value = details.get(key)
                if value is not None:
                    parts.append(f"{key}={value}")
            if details.get("options"):
                parts.append(
                    "options="
                    + json.dumps(details["options"], ensure_ascii=False, sort_keys=True)
                )
            return ", ".join(parts)

        allowed_keys = [
            "interest_rate",
            "saving_period",
            "discount_rate",
            "annual_fee",
            "max_monthly_benefit",
            "max_monthly_amount",
        ]

        parts: list[str] = []

        for key in allowed_keys:
            value = details.get(key)
            if value is None:
                continue
            parts.append(f"{key}={value}")

        return ", ".join(parts)
