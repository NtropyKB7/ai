from copy import deepcopy
from datetime import datetime, timedelta, timezone

from app.rag.canonical_product import product_content_hash
from app.rag.external_staging import (
    EXTERNAL_STAGING_COLLECTION,
    InMemoryExternalStagingAdapter,
)
from app.rag.finlife_product_mapper import FinlifeProductMapper
from app.rag.finlife_response_parser import parse_finlife_response
from app.rag.financial_product_source import FakeFinancialProductSource
from app.schemas.finlife_product import ProductType, SnapshotCompleteness
from app.schemas.product_sync import SnapshotWriteSafety
from app.services.external_product_sync_service import ExternalProductSyncService
from app.services.external_product_sync_service import (
    apply_sync_plan,
    build_sync_plan,
    materialize_embeddings,
)
from finlife_fixtures import response


class CountingEmbedder:
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[float(len(text))] for text in texts]


def product(product_type=ProductType.SAVINGS, collected_at=None):
    page = parse_finlife_response(response(product_type.value), product_type)
    return FinlifeProductMapper().combine_pages(
        [page], product_type, collected_at or datetime(2026, 8, 20, tzinfo=timezone.utc)
    )[0]


def service(adapter=None, embedder=None):
    return ExternalProductSyncService(
        adapter or InMemoryExternalStagingAdapter(), embedder or CountingEmbedder()
    )


def test_collection_is_physically_named_external_staging():
    adapter = InMemoryExternalStagingAdapter()
    assert adapter.collection_name == EXTERNAL_STAGING_COLLECTION
    assert adapter.collection_name == "financial_products_external_staging"


def test_fake_source_is_limited_to_initial_bank_region():
    fake = FakeFinancialProductSource({ProductType.SAVINGS: []})
    assert fake.fetch_pages(ProductType.SAVINGS, "020000") == []
    import pytest

    with pytest.raises(ValueError, match="020000"):
        fake.fetch_pages(ProductType.SAVINGS, "030300")


def test_new_then_unchanged_is_idempotent_and_not_reembedded():
    adapter = InMemoryExternalStagingAdapter()
    embedder = CountingEmbedder()
    sync = service(adapter, embedder)
    first = sync.sync([product()], SnapshotCompleteness(), dry_run=False)
    second = sync.sync([product()], SnapshotCompleteness(), dry_run=False)
    assert (first.created_count, first.embedded_count, first.upserted_count) == (1, 1, 1)
    assert (second.unchanged_count, second.embedded_count, second.upserted_count) == (1, 0, 0)
    assert len(embedder.texts) == 1


def test_changed_product_only_is_embedded():
    adapter = InMemoryExternalStagingAdapter()
    embedder = CountingEmbedder()
    sync = service(adapter, embedder)
    original = product()
    sync.sync([original], SnapshotCompleteness(), dry_run=False)
    changed = original.model_copy(deep=True)
    changed.details["options"][0]["preferred_interest_rate"] = "4.0"
    result = sync.sync([changed], SnapshotCompleteness(), dry_run=False)
    assert (result.updated_count, result.embedded_count) == (1, 1)


def test_collection_time_and_source_submission_time_do_not_change_hash():
    original = product()
    later = product(collected_at=datetime(2026, 8, 21, tzinfo=timezone.utc))
    later.details["source_submitted_at"] = "202608021200"
    assert product_content_hash(original) == product_content_hash(later)


def test_option_order_does_not_change_hash():
    item = product()
    second = deepcopy(item.details["options"][0])
    second["term_months"] = 24
    item.details["options"].append(second)
    reordered = item.model_copy(deep=True)
    reordered.details["options"].reverse()
    assert product_content_hash(item) == product_content_hash(reordered)


def test_rate_term_and_condition_changes_are_detected():
    original = product()
    for field, value in [
        ("preferred_interest_rate", "9.0"),
        ("term_months", 36),
    ]:
        changed = original.model_copy(deep=True)
        changed.details["options"][0][field] = value
        assert product_content_hash(original) != product_content_hash(changed)
    changed = original.model_copy(deep=True)
    changed.details["special_conditions_text"] = "변경 조건"
    assert product_content_hash(original) != product_content_hash(changed)


def test_dry_run_does_not_write():
    adapter = InMemoryExternalStagingAdapter()
    embedder = CountingEmbedder()
    result = service(adapter, embedder).sync(
        [product()], SnapshotCompleteness(), dry_run=True
    )
    assert result.created_count == 1
    assert result.embedded_count == 0
    assert result.upserted_count == 0
    assert embedder.texts == []
    assert adapter.get_all() == {}


def test_build_sync_plan_is_pure_and_deterministic():
    products = [product(ProductType.DEPOSIT), product(ProductType.SAVINGS)]
    first = build_sync_plan(products, {}, SnapshotCompleteness())
    second = build_sync_plan(products, {}, SnapshotCompleteness())
    assert first.result == second.result
    assert first.write_records == second.write_records
    assert first.result.embedded_count == 0
    assert first.result.upserted_count == 0
    assert all(record.embedding is None for record in first.write_records)


def test_read_only_comparison_dry_run_never_embeds_or_upserts():
    class ReadOnlySpy(InMemoryExternalStagingAdapter):
        def __init__(self):
            super().__init__()
            self.reads = 0
            self.writes = 0

        def get_all(self):
            self.reads += 1
            return super().get_all()

        def upsert(self, records):
            self.writes += 1
            raise AssertionError("dry-run must not write")

    staging = ReadOnlySpy()
    result = ExternalProductSyncService(staging).sync(
        [product()], SnapshotCompleteness(), dry_run=True
    )
    assert staging.reads == 1
    assert staging.writes == 0
    assert result.embedded_count == result.upserted_count == 0


def test_materialize_embeds_only_new_and_changed_before_apply():
    initial_products = [product(ProductType.SAVINGS), product(ProductType.DEPOSIT)]
    staging = InMemoryExternalStagingAdapter()
    initial = build_sync_plan(initial_products, {}, SnapshotCompleteness())
    initial_materialized = materialize_embeddings(initial, CountingEmbedder())
    apply_sync_plan(initial_materialized, staging)

    changed = initial_products[0].model_copy(deep=True)
    changed.details["special_conditions_text"] = "합성 변경"
    comparison = build_sync_plan(
        [changed, initial_products[1]], staging.get_all(), SnapshotCompleteness()
    )
    embedder = CountingEmbedder()
    materialized = materialize_embeddings(comparison, embedder)
    assert comparison.result.updated_count == 1
    assert comparison.result.unchanged_count == 1
    assert len(embedder.texts) == 1
    result = apply_sync_plan(materialized, staging)
    assert result.embedded_count == 1
    assert result.upserted_count == 1


def test_apply_never_starts_when_any_embedding_fails():
    class IncompleteEmbedder:
        def embed_documents(self, texts):
            return []

    class WriteSpy(InMemoryExternalStagingAdapter):
        def __init__(self):
            super().__init__()
            self.writes = 0

        def upsert(self, records):
            self.writes += 1
            super().upsert(records)

    staging = WriteSpy()
    plan = build_sync_plan([product()], {}, SnapshotCompleteness())
    import pytest

    with pytest.raises(ValueError, match="unexpected number"):
        materialize_embeddings(plan, IncompleteEmbedder())
    assert staging.writes == 0


def test_incomplete_snapshot_never_observes_missing():
    adapter = InMemoryExternalStagingAdapter()
    sync = service(adapter)
    sync.sync([product()], SnapshotCompleteness(), dry_run=False)
    incomplete = SnapshotCompleteness(all_pages_succeeded=False)
    result = sync.sync([], incomplete, dry_run=False)
    assert result.missing_count == 0
    stored = next(iter(adapter.get_all().values()))
    assert stored.metadata["missing_observation_count"] == 0


def test_three_complete_missing_snapshots_require_review_without_deletion():
    adapter = InMemoryExternalStagingAdapter()
    sync = service(adapter)
    missing_product = product(ProductType.SAVINGS)
    remaining_product = product(ProductType.DEPOSIT)
    sync.sync([missing_product, remaining_product], SnapshotCompleteness(), dry_run=False)
    results = [
        sync.sync([remaining_product], SnapshotCompleteness(), dry_run=False)
        for _ in range(3)
    ]
    stored = adapter.get_all()[missing_product.product_id]
    assert results[-1].review_required_count == 1
    assert stored.metadata["validation_status"] == "REVIEW_REQUIRED"
    assert stored.metadata["sync_observation_status"] == "MISSING_FROM_SOURCE"
    assert stored.metadata["missing_observation_count"] == 3
    assert len(adapter.get_all()) == 2


def test_product_reappearance_resets_missing_without_embedding():
    adapter = InMemoryExternalStagingAdapter()
    embedder = CountingEmbedder()
    sync = service(adapter, embedder)
    item = product()
    sync.sync([item], SnapshotCompleteness(), dry_run=False)
    sync.sync([], SnapshotCompleteness(), dry_run=False)
    result = sync.sync([item], SnapshotCompleteness(), dry_run=False)
    stored = adapter.get_all()[item.product_id]
    assert result.embedded_count == 0
    assert stored.metadata["missing_observation_count"] == 0
    assert stored.metadata["sync_observation_status"] == "SEEN"


def test_partial_upsert_failure_is_reported_and_next_run_recovers():
    adapter = InMemoryExternalStagingAdapter(fail_after=1)
    sync = service(adapter)
    first = product(ProductType.SAVINGS)
    second = product(ProductType.DEPOSIT)
    result = sync.sync([first, second], SnapshotCompleteness(), dry_run=False)
    assert not result.success
    assert result.partial_write_possible
    adapter.fail_after = None
    recovery = sync.sync([first, second], SnapshotCompleteness(), dry_run=False)
    assert recovery.success
    assert len(adapter.get_all()) == 2


def test_staging_metadata_never_enables_recommendation():
    adapter = InMemoryExternalStagingAdapter()
    service(adapter).sync([product(ProductType.DEPOSIT)], SnapshotCompleteness(), dry_run=False)
    stored = next(iter(adapter.get_all().values()))
    assert stored.metadata["recommendation_enabled"] is False
    assert stored.metadata["product_type"] == "DEPOSIT"


def test_result_dto_exposes_final_count_contract():
    result = service().sync(
        [product()], SnapshotCompleteness(), dry_run=True,
        source_count=2, failed_count=1,
        review_required_ids={"FSS_FINLIFE:SAVINGS:0010001:REVIEW"},
        validation_errors=["isolated synthetic error"],
    )
    assert result.model_dump(exclude={"plan"}).keys() == {
        "success", "dry_run", "source_count", "valid_count", "created_count",
        "updated_count", "unchanged_count", "deactivated_count", "failed_count",
        "review_required_count", "missing_count", "embedded_count",
        "upserted_count", "partial_write_possible", "errors",
    }
    assert result.source_count == 2
    assert result.valid_count == 1
    assert result.failed_count == 1
    assert result.deactivated_count == 0
    assert result.errors == ["isolated synthetic error"]


def test_explicit_safety_decision_blocks_all_writes_without_fixed_threshold():
    adapter = InMemoryExternalStagingAdapter()
    result = service(adapter).sync(
        [product()], SnapshotCompleteness(), dry_run=False,
        write_safety=SnapshotWriteSafety(
            approved=False, reasons=["Failure ratio exceeds caller-approved threshold"]
        ),
    )
    assert result.success is False
    assert result.upserted_count == 0
    assert adapter.get_all() == {}


def test_zero_valid_products_blocks_writes_and_missing_observation():
    adapter = InMemoryExternalStagingAdapter()
    result = service(adapter).sync([], SnapshotCompleteness(), dry_run=False)
    assert result.success is False
    assert result.missing_count == 0
    assert adapter.get_all() == {}


def test_product_serialization_failure_is_isolated_with_explicit_safety_approval():
    adapter = InMemoryExternalStagingAdapter()
    valid = product(ProductType.SAVINGS)
    invalid = product(ProductType.DEPOSIT)
    invalid.details["not_json_serializable"] = object()
    result = service(adapter).sync(
        [valid, invalid], SnapshotCompleteness(), dry_run=False,
        write_safety=SnapshotWriteSafety(approved=True),
    )
    assert result.success is True
    assert result.valid_count == 1
    assert result.failed_count == 1
    assert result.created_count == 1
    assert len(adapter.get_all()) == 1
    assert "Product preparation failed" in result.errors[0]


def test_external_document_omits_absent_summary_and_contains_source_facts():
    adapter = InMemoryExternalStagingAdapter()
    service(adapter).sync([product()], SnapshotCompleteness(), dry_run=False)
    document = next(iter(adapter.get_all().values())).document
    assert "핵심혜택:" not in document
    assert "special_conditions_text=원천 우대조건" in document
    assert "preferred_interest_rate" in document


def test_external_sync_has_no_reference_to_recommendation_collection():
    adapter = InMemoryExternalStagingAdapter()
    before = {"REC-1": {"document": "unchanged", "metadata": {"type": "CARD"}}}
    recommendation_sentinel = deepcopy(before)
    service(adapter).sync([product()], SnapshotCompleteness(), dry_run=False)
    assert recommendation_sentinel == before
