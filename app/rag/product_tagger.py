from app.schemas.product_knowledge import NormalizedFinancialProduct


class ProductTagger:
    """
    금융상품 검색/추천용 태그를 생성하는 클래스입니다.

    태그 목적:
    - 임베딩만으로 부족한 구조적 검색 보완
    - 상품 유형 / 대상군 / 혜택 축 / 소비 연관성 축 분리
    - 나중에 군집 비교나 메타데이터 필터링의 기반 제공
    """

    def build_tags(
        self,
        product: NormalizedFinancialProduct,
    ) -> list[str]:
        """
        상품 속성 기반 태그 목록을 생성합니다.
        """
        tags: set[str] = set()

        # ------------------------------------------------------------------
        # 1. 상품 유형 태그
        # ------------------------------------------------------------------
        if product.product_type:
            tags.add(f"type:{product.product_type.lower()}")

        # ------------------------------------------------------------------
        # 2. 대상 고객군 태그
        # ------------------------------------------------------------------
        if product.target_group:
            tags.add(f"target:{product.target_group.lower()}")

        # ------------------------------------------------------------------
        # 3. 요약/팁/상세정보 기반 혜택 태그
        # ------------------------------------------------------------------
        joined_text = " ".join(
            [
                product.product_name,
                product.summary,
                product.target_group,
                product.njob_trend_tip,
                str(product.details),
            ]
        ).lower()

        # 저축/적금 계열
        if any(keyword in joined_text for keyword in ["적금", "저축", "금리", "이자"]):
            tags.add("benefit:saving")

        # 교통/이동
        if any(keyword in joined_text for keyword in ["교통", "주유", "이동", "대중교통"]):
            tags.add("expense:transportation")

        # 식비
        if any(keyword in joined_text for keyword in ["식비", "외식", "카페", "배달"]):
            tags.add("expense:food")

        # 쇼핑
        if any(keyword in joined_text for keyword in ["쇼핑", "온라인", "오프라인", "커머스"]):
            tags.add("expense:shopping")

        # N잡러 대상
        if any(keyword in joined_text for keyword in ["n잡", "부수입", "라이더", "프리랜서"]):
            tags.add("persona:njob")

        # 단기/장기 목적
        if any(keyword in joined_text for keyword in ["단기", "짧은", "유동성"]):
            tags.add("term:short")

        if any(keyword in joined_text for keyword in ["장기", "만기", "목돈"]):
            tags.add("term:long")

        return sorted(tags)