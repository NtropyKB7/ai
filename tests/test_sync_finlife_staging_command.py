from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.commands.sync_finlife_staging import (
    ISSUE36_STAGING_RELATIVE_PATH,
    _create_apply_adapter,
    _validated_staging_path,
    main,
)
from app.rag.external_staging import InMemoryExternalStagingAdapter
from app.schemas.finlife_product import ProductType
from app.services.finlife_collection_service import (
    FinlifeCollectionOutcome,
    FinlifeCollectionSummary,
    FinlifeProductTypeSummary,
)
from tests.test_external_product_sync import product


class CollectionService:
    def __init__(self, products):
        self.products = products

    def collect_and_transform(self, top_fin_group_no):
        by_type = []
        for product_type in (ProductType.SAVINGS, ProductType.DEPOSIT):
            count = sum(item.product_type == product_type.value for item in self.products)
            by_type.append(FinlifeProductTypeSummary(
                product_type=product_type,
                page_count=1,
                base_count=count,
                option_count=count,
                source_count=count,
                mapped_count=count,
                valid_count=count,
                invalid_count=0,
                review_required_count=0,
                orphan_option_count=0,
                optionless_product_count=0,
                duplicate_removed_count=0,
                snapshot_complete=True,
            ))
        count = len(self.products)
        return FinlifeCollectionOutcome(
            products=self.products,
            summary=FinlifeCollectionSummary(
                success=True,
                source_count=count,
                mapped_count=count,
                valid_count=count,
                invalid_count=0,
                review_required_count=0,
                orphan_option_count=0,
                optionless_product_count=0,
                duplicate_removed_count=0,
                canonical_hash_success_count=count,
                canonical_hash_failure_count=0,
                knowledge_id_duplicate_count=0,
                snapshot_complete=True,
                by_product_type=by_type,
            ),
        )


class Embedder:
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[1.0] for _ in texts]


def products():
    return [
        product(ProductType.SAVINGS),
        product(ProductType.DEPOSIT),
    ]


def test_path_guard_accepts_only_issue36_repository_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.commands.sync_finlife_staging.settings.CHROMA_DB_DIR",
        str(tmp_path / ".chroma"),
    )
    expected = tmp_path / ISSUE36_STAGING_RELATIVE_PATH
    assert _validated_staging_path(expected, tmp_path) == expected.resolve()
    with pytest.raises(ValueError, match="STAGING_PATH_REJECTED"):
        _validated_staging_path(tmp_path / "elsewhere", tmp_path)


def test_dry_run_reads_only_without_embedding_or_apply(capsys):
    staging = InMemoryExternalStagingAdapter()
    calls = {"read": 0, "apply": 0, "embed": 0}

    def loader(path, name):
        calls["read"] += 1
        return None, staging.get_all()

    def apply_factory(path, name):
        calls["apply"] += 1
        return None, staging

    def embedder_factory():
        calls["embed"] += 1
        return Embedder()

    code = main(
        ["--dry-run"],
        collection_service=CollectionService(products()),
        existing_loader=loader,
        apply_adapter_factory=apply_factory,
        embedder_factory=embedder_factory,
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert calls == {"read": 1, "apply": 0, "embed": 0}
    assert result["created_count"] == 2
    assert result["embedded_count"] == result["upserted_count"] == 0


def test_apply_embeds_all_changes_before_creating_adapter(capsys):
    staging = InMemoryExternalStagingAdapter()
    events = []
    embedder = Embedder()

    def loader(path, name):
        return None, staging.get_all()

    def embedder_factory():
        events.append("embedder")
        return embedder

    def apply_factory(path, name):
        assert len(embedder.texts) == 2
        events.append("adapter")
        return None, staging

    code = main(
        ["--apply"],
        collection_service=CollectionService(products()),
        existing_loader=loader,
        apply_adapter_factory=apply_factory,
        embedder_factory=embedder_factory,
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert events == ["embedder", "adapter"]
    assert result["embedded_count"] == result["upserted_count"] == 2
    assert result["staging_count"] == 2


def test_same_input_is_unchanged_and_not_reembedded(capsys):
    staging = InMemoryExternalStagingAdapter()
    service = CollectionService(products())

    def loader(path, name):
        return None, staging.get_all()

    def apply_factory(path, name):
        return None, staging

    assert main(
        ["--apply"],
        collection_service=service,
        existing_loader=loader,
        apply_adapter_factory=apply_factory,
        embedder_factory=Embedder,
    ) == 0
    capsys.readouterr()
    calls = {"embed": 0}

    def forbidden_embedder():
        calls["embed"] += 1
        raise AssertionError("dry-run must not create an embedder")

    assert main(
        ["--dry-run"],
        collection_service=service,
        existing_loader=loader,
        apply_adapter_factory=apply_factory,
        embedder_factory=forbidden_embedder,
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert calls["embed"] == 0
    assert result["created_count"] == result["updated_count"] == 0
    assert result["unchanged_count"] == 2
    assert result["staging_count"] == 2


def test_wrong_collection_is_rejected_before_collection(capsys):
    with pytest.raises(SystemExit):
        main(
            ["--dry-run", "--collection-name", "financial_products"],
            collection_service=CollectionService(products()),
        )


def test_real_temporary_chroma_round_trip(tmp_path):
    _, adapter = _create_apply_adapter(
        tmp_path / "isolated-staging",
        "financial_products_external_staging",
    )
    plan = build_sync_plan_for_test(products(), adapter.get_all())
    materialized = materialize_plan_for_test(plan)
    result = apply_sync_plan_for_test(materialized, adapter)
    assert result.success is True
    assert len(adapter.get_all()) == 2


def build_sync_plan_for_test(items, existing):
    from app.schemas.finlife_product import SnapshotCompleteness
    from app.services.external_product_sync_service import build_sync_plan

    return build_sync_plan(items, existing, SnapshotCompleteness())


def materialize_plan_for_test(plan):
    from app.services.external_product_sync_service import materialize_embeddings

    return materialize_embeddings(plan, Embedder())


def apply_sync_plan_for_test(plan, adapter):
    from app.services.external_product_sync_service import apply_sync_plan

    return apply_sync_plan(plan, adapter)
