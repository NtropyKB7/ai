from datetime import datetime, timezone
import socket

import chromadb
import pytest

from app.rag.external_staging import (
    EXTERNAL_STAGING_COLLECTION,
    ChromaExternalStagingAdapter,
)
from app.rag.finlife_product_mapper import FinlifeProductMapper
from app.rag.finlife_response_parser import parse_finlife_response
from app.schemas.finlife_product import ProductType, SnapshotCompleteness
from app.services.external_product_sync_service import ExternalProductSyncService
from finlife_fixtures import response


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def guarded_connect(sock, address):
        if not isinstance(address, tuple) or address[0] in {"127.0.0.1", "::1", "localhost"}:
            return original_connect(sock, address)
        raise AssertionError(f"External network is forbidden: {address!r}")

    def guarded_create_connection(address, *args, **kwargs):
        if address[0] in {"127.0.0.1", "::1", "localhost"}:
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError(f"External network is forbidden: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


class CountingEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [[float(len(text)), 1.0] for text in texts]


def make_product(product_type=ProductType.SAVINGS):
    page = parse_finlife_response(response(product_type.value), product_type)
    return FinlifeProductMapper().combine_pages(
        [page], product_type, datetime(2026, 8, 20, tzinfo=timezone.utc)
    )[0]


def test_real_chroma_staging_incremental_round_trip(tmp_path):
    db_path = tmp_path / "isolated-chroma"
    assert db_path.name != ".chroma"
    client = chromadb.PersistentClient(path=str(db_path))
    recommendation = client.get_or_create_collection("synthetic_recommendation_sentinel")
    recommendation.upsert(
        ids=["REC-1"], documents=["unchanged"], embeddings=[[0.0, 1.0]],
        metadatas=[{"product_type": "CARD"}],
    )
    recommendation_before = recommendation.get(include=["documents", "metadatas"])
    collection = client.get_or_create_collection(EXTERNAL_STAGING_COLLECTION)
    adapter = ChromaExternalStagingAdapter(collection)
    embedder = CountingEmbedder()
    sync = ExternalProductSyncService(adapter, embedder)

    savings = make_product(ProductType.SAVINGS)
    deposit = make_product(ProductType.DEPOSIT)
    first = sync.sync([savings, deposit], SnapshotCompleteness(), dry_run=False)
    stored = adapter.get_all()
    assert first.created_count == 2
    assert len(stored) == 2
    assert stored[savings.product_id].document
    assert stored[savings.product_id].metadata["product_type"] == "SAVINGS"
    assert stored[deposit.product_id].metadata["product_type"] == "DEPOSIT"
    assert stored[deposit.product_id].metadata["recommendation_enabled"] is False

    second = sync.sync([savings, deposit], SnapshotCompleteness(), dry_run=False)
    assert second.unchanged_count == 2
    assert second.embedded_count == 0
    assert len(adapter.get_all()) == 2
    assert embedder.calls == 2

    changed = savings.model_copy(deep=True)
    changed.details["special_conditions_text"] = "변경된 합성 조건"
    previous_hash = stored[savings.product_id].metadata["content_hash"]
    third = sync.sync([changed, deposit], SnapshotCompleteness(), dry_run=False)
    refreshed = adapter.get_all()[savings.product_id]
    assert third.updated_count == 1
    assert "변경된 합성 조건" in refreshed.document
    assert refreshed.metadata["content_hash"] != previous_hash
    assert len(adapter.get_all()) == 2
    assert recommendation.get(include=["documents", "metadatas"]) == recommendation_before


def test_real_chroma_adapter_rejects_wrong_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "isolated-chroma"))
    wrong = client.get_or_create_collection("not_external_staging")
    with pytest.raises(ValueError, match="non-staging"):
        ChromaExternalStagingAdapter(wrong)


class PartialFailureCollection:
    name = EXTERNAL_STAGING_COLLECTION

    def __init__(self, collection):
        self.collection = collection
        self.fail_once = True

    def get(self, **kwargs):
        return self.collection.get(**kwargs)

    def upsert(self, **kwargs):
        if self.fail_once:
            self.fail_once = False
            first = {key: value[:1] for key, value in kwargs.items()}
            self.collection.upsert(**first)
            raise RuntimeError("synthetic partial Chroma failure")
        self.collection.upsert(**kwargs)


def test_real_chroma_partial_failure_recovers_idempotently(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "isolated-chroma"))
    real = client.get_or_create_collection(EXTERNAL_STAGING_COLLECTION)
    wrapped = PartialFailureCollection(real)
    adapter = ChromaExternalStagingAdapter(wrapped)
    sync = ExternalProductSyncService(adapter, CountingEmbedder())
    products = [make_product(ProductType.SAVINGS), make_product(ProductType.DEPOSIT)]
    failed = sync.sync(products, SnapshotCompleteness(), dry_run=False)
    assert failed.success is False
    assert failed.partial_write_possible is True
    recovered = sync.sync(products, SnapshotCompleteness(), dry_run=False)
    assert recovered.success is True
    assert len(adapter.get_all()) == 2
