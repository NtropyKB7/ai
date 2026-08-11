from typing import Any, Optional

from pydantic import BaseModel, Field


class RawFinancialProduct(BaseModel):
    """
    원천 금융상품 데이터 DTO입니다.

    현재는 seed_products.json에서 읽어온 원시 데이터를 담는 용도이고,
    나중에는 외부 API / 크롤링 / CSV 수집 결과도 이 구조로 맞춰서 받게 됩니다.
    """

    product_id: str = Field(..., description="상품 고유 식별자")
    product_name: str = Field(..., description="금융상품명")
    product_type: str = Field(..., description="상품 유형. 예: CARD, SAVINGS")
    provider: str = Field(..., description="금융사명")
    summary: str = Field(..., description="핵심 혜택 요약")
    target_group: Optional[str] = Field(default=None, description="추천 대상 고객군")
    njob_trend_tip: Optional[str] = Field(default=None, description="N잡 활용 팁")
    details: dict[str, Any] = Field(default_factory=dict, description="상품 상세 정보")


class NormalizedFinancialProduct(BaseModel):
    """
    정제/정규화가 끝난 금융상품 DTO입니다.

    추천 검색과 임베딩 생성을 위해
    필수 필드는 항상 채워진 상태를 목표로 합니다.
    """

    product_id: str = Field(..., description="상품 고유 식별자")
    product_name: str = Field(..., description="정규화된 금융상품명")
    product_type: str = Field(..., description="정규화된 상품 유형")
    provider: str = Field(..., description="정규화된 금융사명")
    summary: str = Field(..., description="정규화된 핵심 혜택 요약")
    target_group: str = Field(default="", description="정규화된 추천 대상 고객군")
    njob_trend_tip: str = Field(default="", description="정규화된 N잡 활용 팁")
    details: dict[str, Any] = Field(default_factory=dict, description="정규화된 상세 정보")

    # 추천/검색 보조 메타데이터
    tags: list[str] = Field(default_factory=list, description="상품 검색/추천용 태그 목록")
    is_active: bool = Field(default=True, description="현재 추천 대상 활성 여부")


class ProductKnowledgeDocument(BaseModel):
    """
    ChromaDB upsert 직전 단계의 문서 DTO입니다.

    임베딩 대상 문서 본문(document),
    메타데이터(metadata),
    문서 식별자(id)를 한 곳에 묶습니다.
    """

    product_id: str = Field(..., description="문서 식별자 = 상품 식별자")
    document_text: str = Field(..., description="임베딩 대상 문서 본문")
    metadata: dict[str, Any] = Field(..., description="ChromaDB 메타데이터")