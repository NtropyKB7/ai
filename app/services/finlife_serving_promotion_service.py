from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from app.schemas.product_sync import StagingRecord
from app.services.personalized_product_scoring_service import PersonalizedProductScoringService


SERVING_PRODUCT_TYPES = {"SAVINGS", "DEPOSIT"}


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class ServingWriter(Protocol):
    def add(self, records: list[StagingRecord]) -> None: ...
    def count(self) -> int: ...


@dataclass
class ServingPromotionSummary:
    success: bool
    dry_run: bool
    source_count: int
    valid_count: int
    invalid_count: int
    savings_count: int
    deposit_count: int
    embedded_count: int = 0
    written_count: int = 0
    errors: dict[str, int] = field(default_factory=dict)


@dataclass
class ServingPromotionPlan:
    records: list[StagingRecord]
    summary: ServingPromotionSummary


class FinlifeServingPromotionService:
    def __init__(self, expected_counts: dict[str, int] | None = None):
        self.selector = PersonalizedProductScoringService()
        self.expected_counts = expected_counts

    def build_plan(self, staging_records: dict[str, StagingRecord]) -> ServingPromotionPlan:
        records = []
        errors: dict[str, int] = {}
        counts = {"SAVINGS": 0, "DEPOSIT": 0}
        seen = set()
        for knowledge_id, record in sorted(staging_records.items()):
            try:
                if knowledge_id in seen:
                    raise ValueError("DUPLICATE_ID")
                seen.add(knowledge_id)
                metadata = record.metadata
                product_type = metadata.get("product_type")
                if product_type not in SERVING_PRODUCT_TYPES:
                    raise ValueError("UNSUPPORTED_PRODUCT_TYPE")
                if metadata.get("validation_status") != "VALID":
                    raise ValueError("NOT_VALID")
                if metadata.get("sync_observation_status") != "SEEN":
                    raise ValueError("NOT_SEEN")
                details = json.loads(metadata.get("details_json") or "{}")
                product = {
                    "product_id": knowledge_id,
                    "product_name": metadata.get("product_name") or "",
                    "product_type": product_type,
                    "provider": metadata.get("provider") or "",
                    "summary": metadata.get("summary") or "",
                    "target_group": metadata.get("target_group"),
                    "njob_trend_tip": metadata.get("njob_trend_tip"),
                    "details": details,
                }
                if not product["product_name"] or not product["provider"]:
                    raise ValueError("MISSING_REQUIRED_FIELD")
                selected = self.selector.select_option(product)
                if selected is None:
                    raise ValueError("INVALID_OPTIONS")
                serving_metadata = dict(metadata)
                serving_metadata.update({
                    "recommendation_enabled": True,
                    "is_active": "true",
                    "details_json": json.dumps(
                        selected.product["details"], ensure_ascii=False, sort_keys=True,
                        default=str,
                    ),
                })
                records.append(StagingRecord(
                    knowledge_id=knowledge_id,
                    document=record.document,
                    metadata=serving_metadata,
                ))
                counts[product_type] += 1
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                kind = str(exc) if str(exc) in {
                    "DUPLICATE_ID", "UNSUPPORTED_PRODUCT_TYPE", "NOT_VALID",
                    "NOT_SEEN", "MISSING_REQUIRED_FIELD", "INVALID_OPTIONS",
                } else "INVALID_RECORD"
                errors[kind] = errors.get(kind, 0) + 1
        if self.expected_counts is not None:
            for kind, expected in self.expected_counts.items():
                if counts.get(kind, 0) != expected:
                    errors[f"{kind}_COUNT_MISMATCH"] = 1
        summary = ServingPromotionSummary(
            success=not errors and bool(records),
            dry_run=True,
            source_count=len(staging_records),
            valid_count=len(records),
            invalid_count=sum(errors.values()),
            savings_count=counts["SAVINGS"],
            deposit_count=counts["DEPOSIT"],
            errors=errors,
        )
        return ServingPromotionPlan(records, summary)

    @staticmethod
    def materialize(plan: ServingPromotionPlan, embedder: DocumentEmbedder) -> ServingPromotionPlan:
        if not plan.summary.success:
            raise ValueError("SERVING_PLAN_INVALID")
        embeddings = embedder.embed_documents([item.document for item in plan.records])
        if len(embeddings) != len(plan.records):
            raise ValueError("EMBEDDING_COUNT_MISMATCH")
        materialized = ServingPromotionPlan(
            records=[item.model_copy(deep=True) for item in plan.records],
            summary=ServingPromotionSummary(**vars(plan.summary)),
        )
        for record, embedding in zip(materialized.records, embeddings):
            record.embedding = embedding
        materialized.summary.embedded_count = len(embeddings)
        return materialized

    @staticmethod
    def apply(materialized: ServingPromotionPlan, writer: ServingWriter) -> ServingPromotionSummary:
        if any(item.embedding is None for item in materialized.records):
            raise ValueError("EMBEDDINGS_NOT_MATERIALIZED")
        writer.add(materialized.records)
        if writer.count() != len(materialized.records):
            raise ValueError("SERVING_COUNT_MISMATCH")
        if hasattr(writer, "validate"):
            writer.validate(materialized.records)
        result = ServingPromotionSummary(**vars(materialized.summary))
        result.dry_run = False
        result.written_count = len(materialized.records)
        return result
