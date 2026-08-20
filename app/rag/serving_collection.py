from __future__ import annotations

from app.schemas.product_sync import StagingRecord


class ChromaServingWriter:
    """Writer for a newly-created, explicitly named versioned collection."""

    def __init__(self, collection):
        self.collection = collection

    def add(self, records: list[StagingRecord]) -> None:
        self.collection.add(
            ids=[item.knowledge_id for item in records],
            documents=[item.document for item in records],
            metadatas=[item.metadata for item in records],
            embeddings=[item.embedding for item in records],
        )

    def count(self) -> int:
        return self.collection.count()

    def validate(self, records: list[StagingRecord]) -> None:
        expected_ids = {item.knowledge_id for item in records}
        stored = self.collection.get(include=["metadatas"])
        if set(stored.get("ids") or []) != expected_ids:
            raise ValueError("SERVING_ID_SET_MISMATCH")
        metadatas = stored.get("metadatas") or []
        if any(item.get("product_type") not in {"SAVINGS", "DEPOSIT"} for item in metadatas):
            raise ValueError("SERVING_PRODUCT_TYPE_MISMATCH")
        if records:
            result = self.collection.query(
                query_embeddings=[records[0].embedding], n_results=1,
                include=["metadatas"],
            )
            if not (result.get("ids") or [[]])[0]:
                raise ValueError("SERVING_SEARCH_VALIDATION_FAILED")
