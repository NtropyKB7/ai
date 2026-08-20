import json

from app.commands.promote_finlife_serving import main
from app.commands.promote_finlife_serving import _load_staging, _new_writer
from app.schemas.product_sync import StagingRecord
from app.services.finlife_serving_promotion_service import FinlifeServingPromotionService


def record(kind="SAVINGS", index=1):
    option = {
        "interest_rate_type": "S", "term_months": 12,
        "base_interest_rate": "4.0", "preferred_interest_rate": "4.2",
    }
    if kind == "SAVINGS":
        option["reserve_type"] = "F"
    details = {"source": "FSS_FINLIFE", "options": [option]}
    item_id = f"FSS_FINLIFE:{kind}:001:{index}"
    return StagingRecord(
        knowledge_id=item_id,
        document=f"합성 {kind} 문서",
        metadata={
            "product_id": item_id, "product_name": f"합성 {kind}",
            "product_type": kind, "provider": "합성은행", "summary": "",
            "validation_status": "VALID", "sync_observation_status": "SEEN",
            "recommendation_enabled": False,
            "details_json": json.dumps(details),
        },
    )


class Embedder:
    def __init__(self): self.calls = 0
    def embed_documents(self, texts):
        self.calls += 1
        return [[float(i), 1.0] for i, _ in enumerate(texts)]


class Writer:
    def __init__(self): self.records = []
    def add(self, records): self.records.extend(records)
    def count(self): return len(self.records)


def test_plan_excludes_card_and_requires_valid_seen():
    records = {"s": record("SAVINGS"), "d": record("DEPOSIT")}
    card = record("SAVINGS", 3)
    card.metadata["product_type"] = "CARD"
    records["c"] = card
    plan = FinlifeServingPromotionService().build_plan(records)
    assert not plan.summary.success
    assert plan.summary.valid_count == 2
    assert plan.summary.errors == {"UNSUPPORTED_PRODUCT_TYPE": 1}


def test_production_plan_requires_exactly_58_savings_and_38_deposits():
    items = [record("SAVINGS", i) for i in range(1, 59)]
    items += [record("DEPOSIT", i) for i in range(1, 39)]
    plan = FinlifeServingPromotionService(
        {"SAVINGS": 58, "DEPOSIT": 38}
    ).build_plan({item.knowledge_id: item for item in items})
    assert plan.summary.success
    assert plan.summary.source_count == plan.summary.valid_count == 96
    assert plan.summary.savings_count == 58
    assert plan.summary.deposit_count == 38


def test_all_embeddings_finish_before_write_and_counts_match():
    records = {item.knowledge_id: item for item in [record("SAVINGS"), record("DEPOSIT")]}
    service = FinlifeServingPromotionService()
    plan = service.build_plan(records)
    assert plan.summary.success
    embedder = Embedder()
    materialized = service.materialize(plan, embedder)
    writer = Writer()
    result = service.apply(materialized, writer)
    assert embedder.calls == 1
    assert result.embedded_count == result.written_count == 2
    assert all(item.metadata["recommendation_enabled"] is True for item in writer.records)


def test_cli_dry_run_never_constructs_embedder_or_writer(tmp_path, capsys):
    records = {item.knowledge_id: item for item in [record("SAVINGS"), record("DEPOSIT")]}
    called = {"embedder": 0, "writer": 0}
    result = main([
        "--staging-path", str(tmp_path / "staging"),
        "--target-path", str(tmp_path / "target"),
        "--target-collection", "financial_products_finlife_vtest",
        "--dry-run",
    ], staging_loader=lambda *_: records,
       embedder_factory=lambda: called.__setitem__("embedder", 1),
       writer_factory=lambda *_: called.__setitem__("writer", 1),
       service_factory=FinlifeServingPromotionService)
    assert result == 0
    assert called == {"embedder": 0, "writer": 0}
    assert json.loads(capsys.readouterr().out)["valid_count"] == 2


def test_cli_apply_creates_writer_only_after_embedding(tmp_path):
    records = {item.knowledge_id: item for item in [record("SAVINGS"), record("DEPOSIT")]}
    events = []
    class OrderedEmbedder(Embedder):
        def embed_documents(self, texts):
            events.append("embedded")
            return super().embed_documents(texts)
    writer = Writer()
    result = main([
        "--staging-path", str(tmp_path / "staging"),
        "--target-path", str(tmp_path / "target"),
        "--target-collection", "financial_products_finlife_vtest",
        "--apply",
    ], staging_loader=lambda *_: records,
       embedder_factory=OrderedEmbedder,
       writer_factory=lambda *_: (events.append("writer") or writer),
       service_factory=FinlifeServingPromotionService)
    assert result == 0
    assert events == ["embedded", "writer"]
    assert writer.count() == 2


def test_temporary_chroma_versioned_serving_integration(tmp_path):
    import chromadb

    staging_path = tmp_path / "staging"
    target_path = tmp_path / "serving"
    source = chromadb.PersistentClient(path=str(staging_path)).create_collection(
        "financial_products_external_staging"
    )
    records = [record("SAVINGS"), record("DEPOSIT")]
    source.add(
        ids=[item.knowledge_id for item in records],
        documents=[item.document for item in records],
        metadatas=[item.metadata for item in records],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    result = main([
        "--staging-path", str(staging_path),
        "--target-path", str(target_path),
        "--target-collection", "financial_products_finlife_vintegration",
        "--apply",
    ], staging_loader=_load_staging, embedder_factory=Embedder,
       writer_factory=_new_writer,
       service_factory=FinlifeServingPromotionService)
    assert result == 0
    target_client = chromadb.PersistentClient(path=str(target_path))
    target = target_client.get_collection("financial_products_finlife_vintegration")
    assert target.count() == 2
    values = target.get(include=["metadatas"])["metadatas"]
    assert {item["product_type"] for item in values} == {"SAVINGS", "DEPOSIT"}
    assert all(item["recommendation_enabled"] is True for item in values)
    assert {item.name for item in target_client.list_collections()} == {
        "financial_products_finlife_vintegration"
    }
