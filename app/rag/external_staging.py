from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from app.schemas.product_sync import StagingRecord


EXTERNAL_STAGING_COLLECTION = "financial_products_external_staging"


class ExternalStagingAdapter(ABC):
    collection_name = EXTERNAL_STAGING_COLLECTION

    @abstractmethod
    def get_all(self) -> dict[str, StagingRecord]: ...

    @abstractmethod
    def upsert(self, records: list[StagingRecord]) -> None: ...


class InMemoryExternalStagingAdapter(ExternalStagingAdapter):
    def __init__(self, records: list[StagingRecord] | None = None, fail_after: int | None = None):
        self._records = {item.knowledge_id: deepcopy(item) for item in records or []}
        self.fail_after = fail_after

    def get_all(self) -> dict[str, StagingRecord]:
        return deepcopy(self._records)

    def upsert(self, records: list[StagingRecord]) -> None:
        for index, record in enumerate(records):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("Synthetic staging failure")
            self._records[record.knowledge_id] = deepcopy(record)


class ChromaExternalStagingAdapter(ExternalStagingAdapter):
    """Adapter over an injected staging collection; it never creates a client."""

    def __init__(self, collection):
        if getattr(collection, "name", None) != self.collection_name:
            raise ValueError("Refusing to use a non-staging Chroma collection")
        self.collection = collection

    def get_all(self) -> dict[str, StagingRecord]:
        result = self.collection.get(include=["documents", "metadatas", "embeddings"])
        ids = result.get("ids") or []
        documents = result.get("documents") or [""] * len(ids)
        metadatas = result.get("metadatas") or [{}] * len(ids)
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = [None] * len(ids)
        return {
            item_id: StagingRecord(
                knowledge_id=item_id,
                document=documents[index],
                metadata=metadatas[index],
                embedding=embeddings[index],
            )
            for index, item_id in enumerate(ids)
        }

    def upsert(self, records: list[StagingRecord]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[item.knowledge_id for item in records],
            documents=[item.document for item in records],
            metadatas=[item.metadata for item in records],
            embeddings=[item.embedding for item in records],
        )
