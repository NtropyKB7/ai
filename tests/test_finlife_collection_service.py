from copy import deepcopy
from datetime import datetime, timezone

import chromadb
import pytest

from app.rag.financial_product_source import FinancialProductSource
from app.rag.finlife_response_parser import parse_finlife_response
from app.schemas.finlife_product import FinlifePage, ProductType
from app.services.finlife_collection_service import (
    FinlifeCollectionError,
    FinlifeCollectionErrorKind,
    FinlifeCollectionService,
)
from finlife_fixtures import response


CAPTURED_AT = datetime(2026, 8, 20, tzinfo=timezone.utc)


def page(product_type):
    return parse_finlife_response(response(product_type.value), product_type)


class Source(FinancialProductSource):
    def __init__(self, pages=None, error_type=None):
        self.pages = pages or {
            ProductType.SAVINGS: [page(ProductType.SAVINGS)],
            ProductType.DEPOSIT: [page(ProductType.DEPOSIT)],
        }
        self.error_type = error_type
        self.calls = []

    def fetch_pages(self, product_type, top_fin_group_no="020000"):
        self.calls.append(product_type)
        if self.error_type == product_type:
            raise RuntimeError("synthetic source failure")
        return deepcopy(self.pages[product_type])


def test_maps_types_independently_then_combines_without_id_collision():
    source = Source()
    outcome = FinlifeCollectionService(source).collect_and_transform(
        collected_at=CAPTURED_AT
    )
    assert source.calls == [ProductType.SAVINGS, ProductType.DEPOSIT]
    assert outcome.summary.success
    assert outcome.summary.mapped_count == 2
    assert outcome.summary.knowledge_id_duplicate_count == 0
    assert {item.product_type for item in outcome.products} == {"SAVINGS", "DEPOSIT"}
    assert len({item.product_id for item in outcome.products}) == 2


def test_one_type_source_failure_stops_combined_result():
    source = Source(error_type=ProductType.DEPOSIT)
    with pytest.raises(FinlifeCollectionError) as error:
        FinlifeCollectionService(source).collect_and_transform()
    assert error.value.kind == FinlifeCollectionErrorKind.SOURCE_COLLECTION_FAILED


def test_incomplete_type_stops_before_mapping_or_other_type():
    incomplete = page(ProductType.SAVINGS)
    incomplete.max_page_no = 2
    source = Source(pages={
        ProductType.SAVINGS: [incomplete],
        ProductType.DEPOSIT: [page(ProductType.DEPOSIT)],
    })
    with pytest.raises(FinlifeCollectionError) as error:
        FinlifeCollectionService(source).collect_and_transform()
    assert error.value.kind == FinlifeCollectionErrorKind.INCOMPLETE_SNAPSHOT
    assert source.calls == [ProductType.SAVINGS]


def test_aggregates_orphan_invalid_optionless_and_duplicates():
    savings = page(ProductType.SAVINGS)
    second_base = savings.bases[0].model_copy(update={"fin_prdt_cd": "P002"})
    savings.bases.append(second_base)
    savings.total_count = 2
    savings.options.append(savings.options[0].model_copy(deep=True))
    savings.options.append(
        savings.options[0].model_copy(update={"fin_prdt_cd": "ORPHAN"})
    )
    source = Source(pages={
        ProductType.SAVINGS: [savings],
        ProductType.DEPOSIT: [page(ProductType.DEPOSIT)],
    })
    outcome = FinlifeCollectionService(source).collect_and_transform(
        collected_at=CAPTURED_AT
    )
    assert outcome.summary.invalid_count == 1
    assert outcome.summary.review_required_count == 1
    assert outcome.summary.orphan_option_count == 1
    assert outcome.summary.optionless_product_count == 1
    assert outcome.summary.duplicate_removed_count == 1


def test_duplicate_knowledge_id_within_type_is_rejected():
    savings = page(ProductType.SAVINGS)
    second_base = savings.bases[0].model_copy(update={"dcls_month": "202607"})
    second_option = savings.options[0].model_copy(update={"dcls_month": "202607"})
    savings.bases.append(second_base)
    savings.options.append(second_option)
    savings.total_count = 2
    source = Source(pages={
        ProductType.SAVINGS: [savings],
        ProductType.DEPOSIT: [page(ProductType.DEPOSIT)],
    })
    outcome = FinlifeCollectionService(source).collect_and_transform()
    assert outcome.summary.success is False
    assert outcome.summary.knowledge_id_duplicate_count == 1
    assert outcome.products == []


def test_hash_failure_is_counted_without_returning_products(monkeypatch):
    monkeypatch.setattr(
        "app.services.finlife_collection_service.product_content_hash",
        lambda product: (_ for _ in ()).throw(ValueError("synthetic")),
    )
    outcome = FinlifeCollectionService(Source()).collect_and_transform()
    assert outcome.summary.canonical_hash_success_count == 0
    assert outcome.summary.canonical_hash_failure_count == 2
    assert outcome.summary.success is False
    assert outcome.products == []


def test_a_path_creates_no_chroma_client_staging_embedding_or_files(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("A path must not create Chroma")
        ),
    )
    before = set(tmp_path.iterdir())
    outcome = FinlifeCollectionService(Source()).collect_and_transform()
    assert outcome.summary.success
    assert set(tmp_path.iterdir()) == before
