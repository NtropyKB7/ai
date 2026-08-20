from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from app.commands.validate_finlife_http import _source
from app.core.config import settings
from app.rag.external_staging import (
    EXTERNAL_STAGING_COLLECTION,
    ChromaExternalStagingAdapter,
)
from app.schemas.finlife_product import SnapshotCompleteness
from app.schemas.product_sync import SnapshotWriteSafety
from app.services.external_product_sync_service import (
    apply_sync_plan,
    build_sync_plan,
    materialize_embeddings,
)
from app.services.finlife_collection_service import (
    FinlifeCollectionError,
    FinlifeCollectionService,
)


ISSUE36_STAGING_RELATIVE_PATH = Path(".local/issue36-external-staging")


def _validated_staging_path(raw_path: Path, repository_root: Path) -> Path:
    expected = (repository_root / ISSUE36_STAGING_RELATIVE_PATH).resolve()
    actual = raw_path.resolve()
    if actual != expected:
        raise ValueError("STAGING_PATH_REJECTED")
    configured = Path(settings.CHROMA_DB_DIR).resolve()
    if actual == configured:
        raise ValueError("OPERATING_CHROMA_PATH_REJECTED")
    return actual


def _validate_collection_name(name: str) -> None:
    if name != EXTERNAL_STAGING_COLLECTION:
        raise ValueError("STAGING_COLLECTION_REJECTED")
    if name == settings.FINANCIAL_PRODUCT_COLLECTION:
        raise ValueError("RECOMMENDATION_COLLECTION_REJECTED")


def _open_existing_staging(path: Path, collection_name: str):
    if not path.exists():
        return None, {}
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        if exc.__class__.__name__ in {"NotFoundError", "InvalidCollectionException"}:
            return client, {}
        raise RuntimeError("STAGING_READ_FAILED") from None
    adapter = ChromaExternalStagingAdapter(collection)
    return client, adapter.get_all()


def _create_apply_adapter(path: Path, collection_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(collection_name)
    return client, ChromaExternalStagingAdapter(collection)


def _local_embedder():
    return HuggingFaceEmbeddings(
        model_name=settings.FINANCIAL_PRODUCT_EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


def _safe_summary(collection_summary, sync_result, staging_count: int) -> dict:
    return {
        "success": sync_result.success,
        "snapshot_complete": collection_summary.snapshot_complete,
        "source_count": sync_result.source_count,
        "valid_count": sync_result.valid_count,
        "failed_count": sync_result.failed_count,
        "review_required_count": sync_result.review_required_count,
        "created_count": sync_result.created_count,
        "updated_count": sync_result.updated_count,
        "unchanged_count": sync_result.unchanged_count,
        "missing_count": sync_result.missing_count,
        "embedded_count": sync_result.embedded_count,
        "upserted_count": sync_result.upserted_count,
        "staging_count": staging_count,
        "by_product_type": [
            {
                "product_type": item.product_type.value,
                "page_count": item.page_count,
                "source_count": item.source_count,
                "valid_count": item.valid_count,
            }
            for item in collection_summary.by_product_type
        ],
        "error_counts": collection_summary.error_counts,
    }


def main(
    argv=None,
    *,
    collection_service=None,
    existing_loader=_open_existing_staging,
    apply_adapter_factory=_create_apply_adapter,
    embedder_factory=_local_embedder,
) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    expected_path = repository_root / ISSUE36_STAGING_RELATIVE_PATH
    parser = argparse.ArgumentParser(
        description="Compare or apply complete Finlife snapshots to isolated staging"
    )
    parser.add_argument("--top-fin-group-no", default="020000")
    parser.add_argument("--staging-path", type=Path, default=expected_path)
    parser.add_argument(
        "--collection-name", default=EXTERNAL_STAGING_COLLECTION
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    try:
        staging_path = _validated_staging_path(args.staging_path, repository_root)
        _validate_collection_name(args.collection_name)
    except ValueError as exc:
        parser.error(str(exc))

    service = collection_service or FinlifeCollectionService(_source())
    try:
        outcome = service.collect_and_transform(args.top_fin_group_no)
    except FinlifeCollectionError as exc:
        kind = exc.source_error_kind or exc.kind.value
        print(json.dumps({
            "success": False,
            "snapshot_complete": False,
            "error_counts": {kind: 1},
        }, sort_keys=True))
        return 1

    summary = outcome.summary
    if (
        not summary.success
        or not summary.snapshot_complete
        or summary.invalid_count
        or summary.review_required_count
        or summary.canonical_hash_failure_count
        or summary.knowledge_id_duplicate_count
    ):
        print(json.dumps({
            "success": False,
            "snapshot_complete": summary.snapshot_complete,
            "error_counts": {"SNAPSHOT_SAFETY_REJECTED": 1},
        }, sort_keys=True))
        return 1

    _, existing = existing_loader(staging_path, args.collection_name)
    plan = build_sync_plan(
        outcome.products,
        existing,
        SnapshotCompleteness(),
        source_count=summary.source_count,
        failed_count=summary.invalid_count,
        review_required_ids=set(),
        write_safety=SnapshotWriteSafety(approved=True),
    )
    if args.dry_run:
        result = plan.result
        print(json.dumps(
            _safe_summary(summary, result, len(existing)), sort_keys=True
        ))
        return 0 if result.success else 1

    try:
        materialized = materialize_embeddings(plan, embedder_factory())
    except Exception:
        print(json.dumps({
            "success": False,
            "snapshot_complete": True,
            "error_counts": {"EMBEDDING_FAILED": 1},
        }, sort_keys=True))
        return 1
    if materialized.result.embedded_count != (
        materialized.result.created_count + materialized.result.updated_count
    ):
        print(json.dumps({
            "success": False,
            "snapshot_complete": True,
            "error_counts": {"EMBEDDING_COUNT_MISMATCH": 1},
        }, sort_keys=True))
        return 1

    _, adapter = apply_adapter_factory(staging_path, args.collection_name)
    result = apply_sync_plan(materialized, adapter)
    staging_count = len(adapter.get_all())
    print(json.dumps(
        _safe_summary(summary, result, staging_count), sort_keys=True
    ))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
