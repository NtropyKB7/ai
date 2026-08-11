import json

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.schemas.product_knowledge import ProductKnowledgeDocument


class ChromaManager:
    """
    ChromaDB 컬렉션과 임베딩 모델을 관리하는 클래스입니다.

    책임:
    - 컬렉션 생성/조회
    - 문서 임베딩 생성
    - ChromaDB upsert/query 수행
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
        금융상품 문서 목록을 임베딩 후 ChromaDB에 upsert합니다.

        Args:
            documents: upsert 대상 금융상품 문서 목록

        Returns:
            upsert된 문서 수
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

    def search_top_product(
        self,
        query_text: str,
    ) -> dict:
        """
        추천 검색용 쿼리 텍스트로 가장 유사한 금융상품 1개를 반환합니다.
        """
        collection = self.get_or_create_collection()

        query_vector = self.embeddings.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
        )

        if not results or not results["metadatas"] or not results["metadatas"][0]:
            raise ValueError("ChromaDB에서 일치하는 금융상품을 찾지 못했습니다.")

        top_metadata = results["metadatas"][0][0]
        details_dict = json.loads(top_metadata.get("details_json", "{}"))

        return {
            "product_id": top_metadata["product_id"],
            "product_name": top_metadata["product_name"],
            "product_type": top_metadata["product_type"],
            "provider": top_metadata["provider"],
            "summary": top_metadata["summary"],
            "target_group": top_metadata.get("target_group"),
            "njob_trend_tip": top_metadata.get("njob_trend_tip"),
            "details": details_dict,
        }


chroma_manager = ChromaManager()