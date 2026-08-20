import json

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.schemas.product_knowledge import ProductKnowledgeDocument


class ChromaManager:
    """
    ChromaDB 컬렉션과 임베딩 모델을 관리하는 클래스입니다.

    역할:
    - 금융상품 문서 upsert
    - 쿼리 임베딩 생성
    - 유사 상품 후보 검색
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.FINANCIAL_PRODUCT_EMBEDDING_MODEL
        )
        self.collection_name = settings.FINANCIAL_PRODUCT_COLLECTION

    def heartbeat(self):
        """
        ChromaDB 연결 상태를 확인합니다.
        """
        return self.client.heartbeat()

    def get_or_create_collection(self):
        """
        금융상품 컬렉션이 없으면 생성하고, 있으면 기존 컬렉션을 반환합니다.
        """
        return self.client.get_or_create_collection(name=self.collection_name)

    def upsert_product_documents(
        self,
        documents: list[ProductKnowledgeDocument],
    ) -> int:
        """
        금융상품 문서 목록을 임베딩하여 ChromaDB에 upsert합니다.
        """
        if not documents:
            return 0

        collection = self.get_or_create_collection()

        ids = [item.product_id for item in documents]
        doc_texts = [item.document_text for item in documents]
        metadatas = [item.metadata for item in documents]
        embeddings = self.embeddings.embed_documents(doc_texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=doc_texts,
            metadatas=metadatas,
        )

        return len(ids)

    def search_products(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        추천 검색용 쿼리 텍스트로 유사한 금융상품 후보 목록을 반환합니다.

        중요:
        - RAG는 여기서 최종 추천을 결정하지 않습니다.
        - 이 메서드는 "후보 검색"까지만 담당합니다.
        - 최종 선택은 recommendation_service의 재점수화 로직이 맡습니다.
        """
        collection = self.get_or_create_collection()
        query_vector = self.embeddings.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
        )

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            raise ValueError("ChromaDB에서 일치하는 금융상품 후보를 찾지 못했습니다.")

        return [
            self._metadata_to_product(metadata)
            for metadata in results["metadatas"][0]
        ]

    def search_top_product(
        self,
        query_text: str,
    ) -> dict:
        """
        기존 단건 검색 호출부와의 호환을 위한 메서드입니다.
        """
        return self.search_products(query_text=query_text, n_results=1)[0]

    def search_finlife_products_with_scores(
        self,
        query_text: str,
    ) -> tuple[list[dict], dict[str, float]]:
        """Return every Finlife saving/deposit plus its cosine similarity.

        Issue #38 serving collections are created with cosine space. Vector
        order never limits eligibility: all records are deterministically
        rescored by the recommendation service.
        """
        collection = self.get_or_create_collection()
        count = collection.count()
        if count <= 0:
            raise ValueError("ChromaDB에서 일치하는 금융상품 후보를 찾지 못했습니다.")
        query_vector = self.embeddings.embed_query(query_text)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=count,
            include=["metadatas", "distances"],
        )
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        pairs = []
        for metadata, distance in zip(metadatas, distances):
            if metadata.get("product_type") not in {"SAVINGS", "DEPOSIT"}:
                continue
            if metadata.get("validation_status", "VALID") != "VALID":
                continue
            if metadata.get("sync_observation_status", "SEEN") != "SEEN":
                continue
            product = self._metadata_to_product(metadata)
            pairs.append((product, 1.0 - float(distance)))
        pairs.sort(key=lambda pair: pair[0]["product_id"])
        if not pairs:
            raise ValueError("ChromaDB에서 일치하는 금융상품 후보를 찾지 못했습니다.")
        return (
            [pair[0] for pair in pairs],
            {pair[0]["product_id"]: pair[1] for pair in pairs},
        )

    def _metadata_to_product(
        self,
        metadata: dict,
    ) -> dict:
        """
        ChromaDB metadata를 추천 로직에서 쓰기 쉬운 dict로 변환합니다.
        """
        details_dict = json.loads(metadata.get("details_json", "{}"))

        raw_tags = metadata.get("tags", [])
        if isinstance(raw_tags, str):
            try:
                tags = json.loads(raw_tags)
            except json.JSONDecodeError:
                tags = [raw_tags]
        elif isinstance(raw_tags, list):
            tags = raw_tags
        else:
            tags = []

        return {
            "product_id": metadata["product_id"],
            "product_name": metadata["product_name"],
            "product_type": metadata["product_type"],
            "provider": metadata["provider"],
            "summary": metadata["summary"],
            "target_group": metadata.get("target_group"),
            "njob_trend_tip": metadata.get("njob_trend_tip"),
            "details": details_dict,
            "tags": tags,
        }


chroma_manager = ChromaManager()
