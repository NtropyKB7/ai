from __future__ import annotations

import json
from copy import deepcopy
from typing import Protocol

from app.rag.canonical_product import product_content_hash
from app.rag.external_staging import ExternalStagingAdapter
from app.rag.product_document_builder import ProductDocumentBuilder
from app.rag.product_normalizer import ProductNormalizer
from app.rag.product_tagger import ProductTagger
from app.schemas.finlife_product import SnapshotCompleteness
from app.schemas.product_knowledge import RawFinancialProduct
from app.schemas.product_sync import (
    ChangeType,
    ProductSyncPlanItem,
    ProductSyncResult,
    StagingRecord,
    SnapshotWriteSafety,
    SyncObservationStatus,
    ValidationStatus,
)


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class ExternalProductSyncService:
    def __init__(self, staging: ExternalStagingAdapter, embedder: DocumentEmbedder):
        self.staging = staging
        self.embedder = embedder
        self.normalizer = ProductNormalizer()
        self.tagger = ProductTagger()
        self.document_builder = ProductDocumentBuilder()

    def sync(
        self,
        products: list[RawFinancialProduct],
        completeness: SnapshotCompleteness,
        dry_run: bool = True,
        source_count: int | None = None,
        failed_count: int = 0,
        review_required_ids: set[str] | None = None,
        write_safety: SnapshotWriteSafety | None = None,
        validation_errors: list[str] | None = None,
    ) -> ProductSyncResult:
        self._validate_input(products)
        existing = self.staging.get_all()
        planned_records: list[StagingRecord] = []
        plan: list[ProductSyncPlanItem] = []
        changed_documents = []
        isolated_errors = list(validation_errors or [])
        runtime_failed_count = 0

        for product in sorted(products, key=lambda item: item.product_id):
            try:
                content_hash = product_content_hash(product)
                previous = existing.get(product.product_id)
                if previous is None:
                    change = ChangeType.NEW
                elif previous.metadata.get("content_hash") != content_hash:
                    change = ChangeType.CHANGED
                else:
                    change = ChangeType.UNCHANGED
                if change != ChangeType.UNCHANGED:
                    document = self._build_record(product, content_hash)
                    planned_records.append(document)
                    changed_documents.append(document)
                elif self._needs_seen_reset(previous):
                    restored = deepcopy(previous)
                    restored.metadata["sync_observation_status"] = SyncObservationStatus.SEEN.value
                    restored.metadata["missing_observation_count"] = 0
                    restored.metadata["validation_status"] = self._base_validation_status(restored)
                    planned_records.append(restored)
                plan.append(ProductSyncPlanItem(
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
        missing_records = []
        if completeness.is_complete and plan and safety.approved:
            for knowledge_id, previous in existing.items():
                if knowledge_id in current_ids:
                    continue
                missing = deepcopy(previous)
                count = int(missing.metadata.get("missing_observation_count", 0)) + 1
                missing.metadata["missing_observation_count"] = count
                missing.metadata["sync_observation_status"] = SyncObservationStatus.MISSING_FROM_SOURCE.value
                if count >= 3:
                    missing.metadata["validation_status"] = ValidationStatus.REVIEW_REQUIRED.value
                missing_records.append(missing)

        to_embed = [record.document for record in changed_documents]
        embeddings = self.embedder.embed_documents(to_embed) if to_embed else []
        if len(embeddings) != len(changed_documents):
            raise ValueError("Embedder returned an unexpected number of vectors")
        for record, embedding in zip(changed_documents, embeddings):
            record.embedding = embedding

        writes = planned_records + missing_records
        operational_errors: list[str] = []
        partial_write_possible = False
        upserted = 0
        write_blockers = []
        if not completeness.is_complete:
            write_blockers.append("Snapshot completeness is not trusted")
        if not plan:
            write_blockers.append("No valid products are available for staging")
        if not safety.approved:
            write_blockers.extend(safety.reasons or ["Snapshot safety was not approved"])
        if not dry_run and write_blockers:
            operational_errors.extend(write_blockers)
        elif not dry_run and writes:
            try:
                self.staging.upsert(writes)
                upserted = len(writes)
            except Exception as exc:
                operational_errors.append(str(exc))
                partial_write_possible = True

        counts = {kind: sum(item.change_type == kind for item in plan) for kind in ChangeType}
        review_ids = set(review_required_ids or set())
        review_ids.update(
            item.knowledge_id
            for item in planned_records
            if item.metadata.get("validation_status") == ValidationStatus.REVIEW_REQUIRED.value
        )
        review_ids.update(
            item.knowledge_id
            for item in missing_records
            if item.metadata.get("validation_status") == ValidationStatus.REVIEW_REQUIRED.value
        )
        return ProductSyncResult(
            success=not operational_errors,
            dry_run=dry_run,
            source_count=source_count if source_count is not None else len(products) + failed_count,
            valid_count=len(plan),
            created_count=counts[ChangeType.NEW],
            updated_count=counts[ChangeType.CHANGED],
            unchanged_count=counts[ChangeType.UNCHANGED],
            deactivated_count=0,
            failed_count=total_failed_count,
            review_required_count=len(review_ids),
            missing_count=len(missing_records),
            embedded_count=len(embeddings),
            upserted_count=upserted,
            partial_write_possible=partial_write_possible,
            errors=[*isolated_errors, *operational_errors],
            plan=plan,
        )

    def _validate_input(self, products):
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
            "details_json": json.dumps(details, ensure_ascii=False, sort_keys=True),
        })
        return StagingRecord(
            knowledge_id=product.product_id,
            document=document.document_text,
            metadata=metadata,
        )

    @staticmethod
    def _needs_seen_reset(record):
        return (
            record.metadata.get("sync_observation_status") != SyncObservationStatus.SEEN.value
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
