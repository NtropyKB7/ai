from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from app.schemas.finlife_product import FinlifePage, ProductType


class FinancialProductSource(ABC):
    @abstractmethod
    def fetch_pages(
        self, product_type: ProductType, top_fin_group_no: str = "020000"
    ) -> list[FinlifePage]:
        """Return already obtained source pages; implementations may not imply HTTP."""


class FakeFinancialProductSource(FinancialProductSource):
    def __init__(self, pages: dict[ProductType, list[FinlifePage]]):
        self._pages = deepcopy(pages)

    def fetch_pages(
        self, product_type: ProductType, top_fin_group_no: str = "020000"
    ) -> list[FinlifePage]:
        if top_fin_group_no != "020000":
            raise ValueError("Source-independent phase supports topFinGrpNo=020000 only")
        return deepcopy(self._pages.get(product_type, []))
