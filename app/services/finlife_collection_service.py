from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.rag.canonical_product import product_content_hash
from app.rag.financial_product_source import FinancialProductSource
from app.rag.finlife_product_mapper import FinlifeProductMapper
from app.rag.snapshot_validator import validate_complete_pages
from app.schemas.finlife_product import ProductType
from app.schemas.product_knowledge import RawFinancialProduct
from app.schemas.product_sync import MappingIssueKind


class FinlifeCollectionErrorKind(str, Enum):
    SOURCE_COLLECTION_FAILED = "SOURCE_COLLECTION_FAILED"
    INCOMPLETE_SNAPSHOT = "INCOMPLETE_SNAPSHOT"
    MAPPING_FAILED = "MAPPING_FAILED"


class FinlifeCollectionError(RuntimeError):
    def __init__(self, kind: FinlifeCollectionErrorKind, source_error_kind: str | None = None):
        self.kind = kind
        self.source_error_kind = source_error_kind
        super().__init__(f"Finlife collection failed ({kind.value})")


class FinlifeProductTypeSummary(BaseModel):
    product_type: ProductType
    page_count: int
    base_count: int
    option_count: int
    source_count: int
    mapped_count: int
    valid_count: int
    invalid_count: int
    review_required_count: int
    orphan_option_count: int
    optionless_product_count: int
    duplicate_removed_count: int
    snapshot_complete: bool
    issue_counts: dict[str, int] = Field(default_factory=dict)


class FinlifeCollectionSummary(BaseModel):
    success: bool
    source_count: int
    mapped_count: int
    valid_count: int
    invalid_count: int
    review_required_count: int
    orphan_option_count: int
    optionless_product_count: int
    duplicate_removed_count: int
    canonical_hash_success_count: int
    canonical_hash_failure_count: int
    knowledge_id_duplicate_count: int
    snapshot_complete: bool
    by_product_type: list[FinlifeProductTypeSummary]
    error_counts: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class FinlifeCollectionOutcome:
    products: list[RawFinancialProduct]
    summary: FinlifeCollectionSummary


class FinlifeCollectionService:
    """Collect and map complete Finlife snapshots without storage or embedding."""

    PRODUCT_TYPES = (ProductType.SAVINGS, ProductType.DEPOSIT)

    def __init__(
        self,
        source: FinancialProductSource,
        mapper: FinlifeProductMapper | None = None,
    ):
        self.source = source
        self.mapper = mapper or FinlifeProductMapper()

    def collect_and_transform(
        self,
        top_fin_group_no: str = "020000",
        collected_at: datetime | None = None,
    ) -> FinlifeCollectionOutcome:
        captured_at = collected_at or datetime.now(timezone.utc)
        products: list[RawFinancialProduct] = []
        type_summaries: list[FinlifeProductTypeSummary] = []
        aggregate_issue_counts: dict[str, int] = {}

        for product_type in self.PRODUCT_TYPES:
            try:
                pages = self.source.fetch_pages(product_type, top_fin_group_no)
            except Exception as exc:
                source_kind = getattr(getattr(exc, "kind", None), "value", None)
                raise FinlifeCollectionError(
                    FinlifeCollectionErrorKind.SOURCE_COLLECTION_FAILED,
                    source_error_kind=source_kind,
                ) from None

            completeness = validate_complete_pages(pages)
            if not completeness.is_complete:
                raise FinlifeCollectionError(
                    FinlifeCollectionErrorKind.INCOMPLETE_SNAPSHOT
                )
            try:
                mapped = self.mapper.combine_pages_with_issues(
                    pages, product_type, captured_at
                )
            except Exception:
                raise FinlifeCollectionError(
                    FinlifeCollectionErrorKind.MAPPING_FAILED
                ) from None

            base_rows = [row for page in pages for row in page.bases]
            option_rows = [row for page in pages for row in page.options]
            duplicate_removed = (
                len(base_rows) - len({row.model_dump_json() for row in base_rows})
                + len(option_rows) - len({row.model_dump_json() for row in option_rows})
            )
            issue_counts: dict[str, int] = {}
            for issue in mapped.issues:
                key = issue.issue_kind.value
                issue_counts[key] = issue_counts.get(key, 0) + 1
                aggregate_issue_counts[key] = aggregate_issue_counts.get(key, 0) + 1
            type_summaries.append(FinlifeProductTypeSummary(
                product_type=product_type,
                page_count=len(pages),
                base_count=len(base_rows),
                option_count=len(option_rows),
                source_count=len(base_rows),
                mapped_count=len(mapped.products),
                valid_count=len(mapped.products) - mapped.review_required_count,
                invalid_count=mapped.failed_count,
                review_required_count=mapped.review_required_count,
                orphan_option_count=sum(
                    issue.issue_kind == MappingIssueKind.ORPHAN_OPTION
                    for issue in mapped.issues
                ),
                optionless_product_count=sum(
                    issue.issue_kind == MappingIssueKind.OPTIONS_MISSING
                    for issue in mapped.issues
                ),
                duplicate_removed_count=duplicate_removed,
                snapshot_complete=True,
                issue_counts=issue_counts,
            ))
            products.extend(mapped.products)

        ids = [item.product_id for item in products]
        duplicate_id_count = len(ids) - len(set(ids))
        hash_success = 0
        hash_failure = 0
        for product in products:
            try:
                product_content_hash(product)
                hash_success += 1
            except Exception:
                hash_failure += 1

        error_counts = dict(sorted(aggregate_issue_counts.items()))
        if duplicate_id_count:
            error_counts["DUPLICATE_KNOWLEDGE_ID"] = duplicate_id_count
        if hash_failure:
            error_counts["CANONICAL_HASH_FAILED"] = hash_failure
        success = duplicate_id_count == 0 and hash_failure == 0
        summary = FinlifeCollectionSummary(
            success=success,
            source_count=sum(item.source_count for item in type_summaries),
            mapped_count=len(products),
            valid_count=sum(item.valid_count for item in type_summaries),
            invalid_count=sum(item.invalid_count for item in type_summaries),
            review_required_count=sum(
                item.review_required_count for item in type_summaries
            ),
            orphan_option_count=sum(
                item.orphan_option_count for item in type_summaries
            ),
            optionless_product_count=sum(
                item.optionless_product_count for item in type_summaries
            ),
            duplicate_removed_count=sum(
                item.duplicate_removed_count for item in type_summaries
            ),
            canonical_hash_success_count=hash_success,
            canonical_hash_failure_count=hash_failure,
            knowledge_id_duplicate_count=duplicate_id_count,
            snapshot_complete=True,
            by_product_type=type_summaries,
            error_counts=error_counts,
        )
        return FinlifeCollectionOutcome(
            products=products if success else [],
            summary=summary,
        )
