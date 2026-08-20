from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from app.rag.canonical_product import product_content_hash
from app.rag.product_document_builder import ProductDocumentBuilder
from app.rag.product_normalizer import ProductNormalizer
from app.rag.product_tagger import ProductTagger
from app.schemas.finlife_product import SnapshotCompleteness
from app.schemas.product_knowledge import RawFinancialProduct
from app.schemas.product_sync import (
    ChangeType,
    ProductSyncPlanItem,
    ProductSyncResult,
    SnapshotWriteSafety,
    StagingRecord,
    SyncObservationStatus,
    ValidationStatus,
)


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class StagingReader(Protocol):
    def get_all(self) -> dict[str, StagingRecord]: ...


class StagingWriter(Protocol):
    def upsert(self, records: list[StagingRecord]) -> None: ...


@dataclass
class ExternalSyncPlan:
    """In-memory deterministic plan; it performs neither I/O nor embedding."""

    result: ProductSyncResult
    write_records: list[StagingRecord]
    embedding_record_ids: tuple[str, ...]
    write_blockers: tuple[str, ...]


class _SyncPlanBuilder:
    def __init__(self):
        self.normalizer = ProductNormalizer()
        self.tagger = ProductTagger()
        self.document_builder = ProductDocumentBuilder()

    def build(
        self,
        products: list[RawFinancialProduct],
        existing_records: dict[str, StagingRecord],
        completeness: SnapshotCompleteness,
        *,
        source_count: int | None = None,
        failed_count: int = 0,
        review_required_ids: set[str] | None = None,
        write_safety: SnapshotWriteSafety | None = None,
        validation_errors: list[str] | None = None,
    ) -> ExternalSyncPlan:
        self._validate_input(products)
        planned_records: list[StagingRecord] = []
        plan_items: list[ProductSyncPlanItem] = []
        embedding_ids: list[str] = []
        isolated_errors = list(validation_errors or [])
        runtime_failed_count = 0

        for product in sorted(products, key=lambda item: item.product_id):
            try:
                content_hash = product_content_hash(product)
                previous = existing_records.get(product.product_id)
                if previous is None:
                    change = ChangeType.NEW
                elif previous.metadata.get("content_hash") != content_hash:
                    change = ChangeType.CHANGED
                else:
                    change = ChangeType.UNCHANGED
                if change != ChangeType.UNCHANGED:
                    planned_records.append(self._build_record(product, content_hash))
                    embedding_ids.append(product.product_id)
                elif self._needs_seen_reset(previous):
                    restored = deepcopy(previous)
                    restored.metadata["sync_observation_status"] = (
                        SyncObservationStatus.SEEN.value
                    )
                    restored.metadata["missing_observation_count"] = 0
                    restored.metadata["validation_status"] = (
                        self._base_validation_status(restored)
                    )
                    planned_records.append(restored)
                plan_items.append(ProductSyncPlanItem(
                    knowledge_id=product.product_id,
                    change_type=change,
                    content_hash=content_hash,
                ))
            except Exception as exc:
                runtime_failed_count += 1
                isolated_errors.append(
                    f"Product preparation failed: {product.product_id}: {exc}"
                )

        total_failed_count = failed_count + runtime_failed_count
        safety = write_safety or SnapshotWriteSafety(
            approved=total_failed_count == 0,
            reasons=(
                []
                if total_failed_count == 0
                else ["Explicit safety approval is required when products are isolated"]
            ),
        )

        current_ids = {product.product_id for product in products}
        missing_records: list[StagingRecord] = []
        if completeness.is_complete and plan_items and safety.approved:
            for knowledge_id, previous in sorted(existing_records.items()):
                if knowledge_id in current_ids:
                    continue
                missing = deepcopy(previous)
                count = int(missing.metadata.get("missing_observation_count", 0)) + 1
                missing.metadata["missing_observation_count"] = count
                missing.metadata["sync_observation_status"] = (
                    SyncObservationStatus.MISSING_FROM_SOURCE.value
                )
                if count >= 3:
                    missing.metadata["validation_status"] = (
                        ValidationStatus.REVIEW_REQUIRED.value
                    )
                missing_records.append(missing)

        write_records = planned_records + missing_records
        blockers = []
        if not completeness.is_complete:
            blockers.append("Snapshot completeness is not trusted")
        if not plan_items:
            blockers.append("No valid products are available for staging")
        if not safety.approved:
            blockers.extend(safety.reasons or ["Snapshot safety was not approved"])

        counts = {
            kind: sum(item.change_type == kind for item in plan_items)
            for kind in ChangeType
        }
        review_ids = set(review_required_ids or set())
        review_ids.update(
            item.knowledge_id
            for item in planned_records
            if item.metadata.get("validation_status")
            == ValidationStatus.REVIEW_REQUIRED.value
        )
        review_ids.update(
            item.knowledge_id
            for item in missing_records
            if item.metadata.get("validation_status")
            == ValidationStatus.REVIEW_REQUIRED.value
        )
        result = ProductSyncResult(
            success=True,
            dry_run=True,
            source_count=(
                source_count
                if source_count is not None
                else len(products) + failed_count
            ),
            valid_count=len(plan_items),
            created_count=counts[ChangeType.NEW],
            updated_count=counts[ChangeType.CHANGED],
            unchanged_count=counts[ChangeType.UNCHANGED],
            deactivated_count=0,
            failed_count=total_failed_count,
            review_required_count=len(review_ids),
            missing_count=len(missing_records),
            embedded_count=0,
            upserted_count=0,
            partial_write_possible=False,
            errors=isolated_errors,
            plan=plan_items,
        )
        return ExternalSyncPlan(
            result=result,
            write_records=write_records,
            embedding_record_ids=tuple(embedding_ids),
            write_blockers=tuple(blockers),
        )

    @staticmethod
    def _validate_input(products):
        ids = [item.product_id for item in products]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate knowledge_id in snapshot")
        for product in products:
            if product.product_type not in {"SAVINGS", "DEPOSIT"}:
                raise ValueError("External staging accepts SAVINGS or DEPOSIT only")
            expected_prefix = f"FSS_FINLIFE:{product.product_type}:"
            if not product.product_id.startswith(expected_prefix):
                raise ValueError("Invalid Finlife knowledge_id")

    def _build_record(self, product, content_hash):
        normalized = self.normalizer.normalize(product)
        normalized.tags = self.tagger.build_tags(normalized)
        document = self.document_builder.build(normalized)
        details = product.details
        metadata = dict(document.metadata)
        metadata.update({
            "knowledge_id": product.product_id,
            "source": "FSS_FINLIFE",
            "source_region_code": details["source_region_code"],
            "source_product_id": details["source_product_id"],
            "source_provider_id": details["source_provider_id"],
            "prdt_div": "S" if product.product_type == "SAVINGS" else "D",
            "content_hash": content_hash,
            "disclosure_month": details["disclosure_month"],
            "disclosure_start_date": details.get("disclosure_start_date") or "",
            "disclosure_end_date": details.get("disclosure_end_date") or "",
            "source_submitted_at": details.get("source_submitted_at") or "",
            "collected_at": details["collected_at"],
            "validation_status": (
                ValidationStatus.VALID.value
                if details.get("options")
                else ValidationStatus.REVIEW_REQUIRED.value
            ),
            "sync_observation_status": SyncObservationStatus.SEEN.value,
            "missing_observation_count": 0,
            "recommendation_enabled": False,
            "details_json": json.dumps(
                details, ensure_ascii=False, sort_keys=True
            ),
        })
        return StagingRecord(
            knowledge_id=product.product_id,
            document=document.document_text,
            metadata=metadata,
        )

    @staticmethod
    def _needs_seen_reset(record):
        return (
            record.metadata.get("sync_observation_status")
            != SyncObservationStatus.SEEN.value
            or int(record.metadata.get("missing_observation_count", 0)) != 0
        )

    @staticmethod
    def _base_validation_status(record):
        details = json.loads(record.metadata.get("details_json", "{}"))
        return (
            ValidationStatus.VALID.value
            if details.get("options")
            else ValidationStatus.REVIEW_REQUIRED.value
        )


def build_sync_plan(
    products: list[RawFinancialProduct],
    existing_records: dict[str, StagingRecord],
    completeness: SnapshotCompleteness,
    **kwargs,
) -> ExternalSyncPlan:
    return _SyncPlanBuilder().build(
        products, existing_records, completeness, **kwargs
    )


def materialize_embeddings(
    plan: ExternalSyncPlan,
    embedder: DocumentEmbedder,
) -> ExternalSyncPlan:
    materialized = deepcopy(plan)
    records = {
        item.knowledge_id: item for item in materialized.write_records
    }
    candidates = [records[item_id] for item_id in materialized.embedding_record_ids]
    embeddings = (
        embedder.embed_documents([item.document for item in candidates])
        if candidates
        else []
    )
    if len(embeddings) != len(candidates):
        raise ValueError("Embedder returned an unexpected number of vectors")
    for record, embedding in zip(candidates, embeddings):
        record.embedding = embedding
    materialized.result.embedded_count = len(embeddings)
    return materialized


def apply_sync_plan(
    plan: ExternalSyncPlan,
    staging: StagingWriter,
) -> ProductSyncResult:
    result = plan.result.model_copy(deep=True)
    result.dry_run = False
    if plan.write_blockers:
        result.success = False
        result.errors.extend(plan.write_blockers)
        return result
    records_by_id = {item.knowledge_id: item for item in plan.write_records}
    if any(
        records_by_id[item_id].embedding is None
        for item_id in plan.embedding_record_ids
    ):
        raise ValueError("Sync plan embeddings must be fully materialized before apply")
    if not plan.write_records:
        return result
    try:
        staging.upsert(plan.write_records)
        result.upserted_count = len(plan.write_records)
    except Exception as exc:
        result.success = False
        result.partial_write_possible = True
        result.errors.append(str(exc))
    return result


class ExternalProductSyncService:
    """Compatibility facade over read, plan, embed, and apply stages."""

    def __init__(
        self,
        staging: StagingReader | StagingWriter,
        embedder: DocumentEmbedder | None = None,
    ):
        self.staging = staging
        self.embedder = embedder

    def sync(
        self,
        products: list[RawFinancialProduct],
        completeness: SnapshotCompleteness,
        dry_run: bool = True,
        **kwargs,
    ) -> ProductSyncResult:
        existing = self.staging.get_all()
        plan = build_sync_plan(products, existing, completeness, **kwargs)
        if dry_run:
            return plan.result
        if plan.write_blockers:
            return apply_sync_plan(plan, self.staging)
        if self.embedder is None:
            raise ValueError("Non-dry-run synchronization requires an embedder")
        materialized = materialize_embeddings(plan, self.embedder)
        return apply_sync_plan(materialized, self.staging)
