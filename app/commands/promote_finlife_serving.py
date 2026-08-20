from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.core.config import settings
from app.rag.external_staging import EXTERNAL_STAGING_COLLECTION, ChromaExternalStagingAdapter
from app.rag.serving_collection import ChromaServingWriter
from app.services.finlife_serving_promotion_service import FinlifeServingPromotionService


VERSIONED_COLLECTION = re.compile(r"^financial_products_finlife_v[0-9A-Za-z_-]+$")


def _validate_paths(staging_path: Path, target_path: Path, target_collection: str):
    staging = staging_path.resolve()
    target = target_path.resolve()
    operating = Path(settings.CHROMA_DB_DIR).resolve()
    if staging == operating or target == operating:
        raise ValueError("OPERATING_CHROMA_PATH_REJECTED")
    if staging == target:
        raise ValueError("STAGING_AND_TARGET_MUST_DIFFER")
    if target_collection == settings.FINANCIAL_PRODUCT_COLLECTION:
        raise ValueError("ACTIVE_COLLECTION_REJECTED")
    if not VERSIONED_COLLECTION.fullmatch(target_collection):
        raise ValueError("VERSIONED_COLLECTION_NAME_REQUIRED")
    return staging, target


def _load_staging(path: Path, collection_name: str):
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(collection_name)
    return ChromaExternalStagingAdapter(collection).get_all()


def _embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.FINANCIAL_PRODUCT_EMBEDDING_MODEL,
        model_kwargs={"local_files_only": True},
    )


def _new_writer(path: Path, name: str):
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    names = {item.name for item in client.list_collections()}
    if name in names:
        raise ValueError("TARGET_COLLECTION_ALREADY_EXISTS")
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "source": "FSS_FINLIFE_ISSUE38"},
    )
    return ChromaServingWriter(collection)


def _summary(value):
    return json.dumps(vars(value), ensure_ascii=False, sort_keys=True)


def main(argv=None, *, staging_loader=_load_staging, embedder_factory=_embedder, writer_factory=_new_writer, service_factory=None):
    parser = argparse.ArgumentParser(description="Validate or build a versioned Finlife serving collection")
    parser.add_argument("--staging-path", type=Path, required=True)
    parser.add_argument("--staging-collection", default=EXTERNAL_STAGING_COLLECTION)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--target-collection", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.staging_collection != EXTERNAL_STAGING_COLLECTION:
        parser.error("STAGING_COLLECTION_REJECTED")
    try:
        staging_path, target_path = _validate_paths(
            args.staging_path, args.target_path, args.target_collection
        )
        records = staging_loader(staging_path, args.staging_collection)
        service = (
            service_factory()
            if service_factory is not None
            else FinlifeServingPromotionService({"SAVINGS": 58, "DEPOSIT": 38})
        )
        plan = service.build_plan(records)
        if args.dry_run:
            print(_summary(plan.summary))
            return 0 if plan.summary.success else 1
        if not plan.summary.success:
            print(_summary(plan.summary))
            return 1
        materialized = service.materialize(plan, embedder_factory())
        # The writer factory creates the collection only after every embedding
        # has succeeded.
        writer = writer_factory(target_path, args.target_collection)
        result = service.apply(materialized, writer)
        print(_summary(result))
        return 0
    except Exception as exc:
        print(json.dumps({"success": False, "error": exc.__class__.__name__}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
