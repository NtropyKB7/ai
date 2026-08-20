from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.rag.external_staging import InMemoryExternalStagingAdapter
from app.rag.finlife_product_mapper import FinlifeProductMapper
from app.rag.finlife_response_parser import parse_finlife_response
from app.rag.snapshot_validator import validate_complete_pages
from app.schemas.finlife_product import ProductType
from app.schemas.product_sync import SnapshotWriteSafety, ValidationStatus
from app.services.external_product_sync_service import ExternalProductSyncService


LOGGER = logging.getLogger("external-product-sync")


class DeterministicLocalEmbedder:
    """Non-semantic local embedder for source-independent validation only."""

    def embed_documents(self, texts):
        return [[float(len(text.encode("utf-8")))] for text in texts]


@contextmanager
def single_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            _lock_file(lock_file)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError("External product synchronization is already running") from exc
        try:
            yield
        finally:
            _unlock_file(lock_file)


def _lock_file(lock_file):
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        lock_file.write("0")
        lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file):
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_fixture(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Fixture must be a list of {product_type, response} objects")
    pages = {ProductType.SAVINGS: [], ProductType.DEPOSIT: []}
    for item in payload:
        product_type = ProductType(item["product_type"])
        pages[product_type].append(parse_finlife_response(item["response"], product_type))
    return pages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate Finlife fixtures and plan staging sync")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/ntropy-finlife-sync.lock"))
    parser.add_argument("--apply-to-fake-staging", action="store_true")
    parser.add_argument("--confirm-source-independent-write", action="store_true")
    parser.add_argument("--approve-isolated-failures", action="store_true")
    args = parser.parse_args(argv)
    if args.apply_to_fake_staging and not args.confirm_source_independent_write:
        parser.error("--apply-to-fake-staging requires --confirm-source-independent-write")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with single_process_lock(args.lock_file):
        pages_by_type = _load_fixture(args.fixture)
        mapper = FinlifeProductMapper()
        products = []
        source_count = 0
        issues = []
        completeness = []
        collected_at = datetime.now(timezone.utc)
        for product_type, pages in pages_by_type.items():
            if not pages:
                continue
            state = validate_complete_pages(pages)
            completeness.append(state)
            mapped = mapper.combine_pages_with_issues(pages, product_type, collected_at)
            products.extend(mapped.products)
            source_count += mapped.source_count
            issues.extend(mapped.issues)
        combined = all(item.is_complete for item in completeness) and bool(completeness)
        if not combined:
            LOGGER.warning("snapshot_incomplete missing_observation=false")
        from app.schemas.finlife_product import SnapshotCompleteness

        isolated_failed_count = sum(
            item.validation_status == ValidationStatus.INVALID for item in issues
        )
        result = ExternalProductSyncService(
            staging=InMemoryExternalStagingAdapter(),
            embedder=DeterministicLocalEmbedder(),
        ).sync(
            products,
            SnapshotCompleteness() if combined else SnapshotCompleteness(all_pages_succeeded=False),
            dry_run=not args.apply_to_fake_staging,
            source_count=source_count,
            failed_count=isolated_failed_count,
            review_required_ids={
                item.knowledge_id
                for item in issues
                if item.knowledge_id
                and item.validation_status == ValidationStatus.REVIEW_REQUIRED
            },
            write_safety=SnapshotWriteSafety(
                approved=(
                    combined
                    and (isolated_failed_count == 0 or args.approve_isolated_failures)
                ),
                reasons=(
                    []
                    if combined and (isolated_failed_count == 0 or args.approve_isolated_failures)
                    else ["Snapshot completeness or isolated-failure safety is not approved"]
                ),
            ),
            validation_errors=[item.message for item in issues],
        )
        summary = result.model_dump(exclude={"plan"})
        LOGGER.info("sync_summary=%s", json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
