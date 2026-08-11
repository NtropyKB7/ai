import json
import os

from app.schemas.product_knowledge import RawFinancialProduct


class ProductSourceLoader:
    """
    금융상품 원천 데이터 로더입니다.

    현재는 로컬 JSON 파일(seed_products.json)을 읽지만,
    나중에는 외부 수집 API / CSV / 크롤링 결과를
    같은 인터페이스로 연결할 수 있도록 분리합니다.
    """

    def load_from_json_file(
        self,
        json_file_path: str,
    ) -> list[RawFinancialProduct]:
        """
        JSON 파일에서 금융상품 원천 데이터를 읽어옵니다.

        Args:
            json_file_path: 원천 데이터 JSON 파일 경로

        Returns:
            RawFinancialProduct 목록
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(
                f"금융상품 원천 데이터 파일을 찾을 수 없습니다: {json_file_path}"
            )

        with open(json_file_path, "r", encoding="utf-8") as file:
            raw_items = json.load(file)

        if not isinstance(raw_items, list):
            raise ValueError("금융상품 원천 데이터 JSON은 리스트 형태여야 합니다.")

        return [
            RawFinancialProduct(**item)
            for item in raw_items
            if isinstance(item, dict)
        ]