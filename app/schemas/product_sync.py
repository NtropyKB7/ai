from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SyncObservationStatus(str, Enum):
    SEEN = "SEEN"
    MISSING_FROM_SOURCE = "MISSING_FROM_SOURCE"


class StagingRecord(BaseModel):
    knowledge_id: str
    document: str
    metadata: dict
    embedding: list[float] | None = None


class ProductSyncPlanItem(BaseModel):
    knowledge_id: str
    change_type: ChangeType
    content_hash: str


class ProductSyncResult(BaseModel):
    success: bool
    dry_run: bool
    source_count: int
    valid_count: int
    created_count: int
    updated_count: int
    unchanged_count: int
    deactivated_count: int = 0
    failed_count: int
    review_required_count: int
    missing_count: int
    embedded_count: int
    upserted_count: int
    partial_write_possible: bool = False
    errors: list[str] = Field(default_factory=list)
    plan: list[ProductSyncPlanItem] = Field(default_factory=list)


class MappingIssue(BaseModel):
    knowledge_id: str | None = None
    validation_status: ValidationStatus
    message: str


class ProductMappingResult(BaseModel):
    source_count: int
    products: list = Field(default_factory=list)
    issues: list[MappingIssue] = Field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return sum(item.validation_status == ValidationStatus.INVALID for item in self.issues)

    @property
    def review_required_count(self) -> int:
        return sum(
            item.validation_status == ValidationStatus.REVIEW_REQUIRED
            for item in self.issues
        )


class SnapshotWriteSafety(BaseModel):
    approved: bool = True
    reasons: list[str] = Field(default_factory=list)
