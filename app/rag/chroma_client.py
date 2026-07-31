import json
import os
import chromadb
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

class ChromaManager:
    """
    1. Chroma Vector DB 및 임베딩 모델 인스턴스를 관리하는 클래스
    - 지식 데이터 인덱싱(Seed) 및 유사도 검색(Retrieval)을 수행합니다.
    """
    def __init__(self):
        # 로컬 디렉토리에 지식 베이스(Vector) 데이터를 영속적으로 저장하기 위한 PersistentClient 설정
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)

        # 2. OpenAI 임베딩 대신 로컬 한국어 모델로 교체
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )
        # 금융상품 지식을 보관할 컬렉션 이름 지정
        self.collection_name = "financial_products"

    def heartbeat(self):
        """
        ChromaDB 커넥션 상태 체크용 핑(ping) 헬스 체크 메서드
        """
        return self.client.heartbeat()

    def get_or_create_collection(self):
        """컬렉션이 없으면 생성하고, 있으면 기존 컬렉션을 가져옴"""
        return self.client.get_or_create_collection(name=self.collection_name)

    def seed_financial_products(self, json_file_path: str = "app/rag/seed_products.json") -> int:
        """
        [초기 지식 적재 프로세스]
        seed_products.json 파일의 데이터를 읽어서 OpenAI 임베딩 후 ChromaDB에 저장합니다.
        """
        # 1. 시드 지식 데이터 파일 존재 여부 검증
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"시드 파일 경로를 찾을 수 없습니다: {json_file_path}")

        with open(json_file_path, "r", encoding="utf-8") as f:
            products = json.load(f)

        collection = self.get_or_create_collection()

        documents, metadatas, ids = [], [], []

        # 2. JSON 객체를 ChromaDB 문서/메타데이터 규격에 맞춰 파싱
        for item in products:
            # 임베딩(Vector) 계산의 대상이 될 본문 문장 (혜택요약 + 타깃군)
            doc_text = f"상품명: {item['product_name']} | 요약: {item['summary']} | 타겟: {item.get('target_group', '')}"
            
            # 검색 결과와 함께 복원할 메타데이터 (details 딕셔너리는 JSON 문자열로 변환하여 보관)
            metadata = {
                "product_id": item["product_id"],
                "product_name": item["product_name"],
                "product_type": item["product_type"],
                "provider": item["provider"],
                "summary": item["summary"],
                "target_group": item.get("target_group", ""),
                "njob_trend_tip": item.get("njob_trend_tip", ""),
                "details_json": json.dumps(item.get("details", {}), ensure_ascii=False)
            }

            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(item["product_id"])

        # 3. OpenAI Embeddings API 호출하여 벡터 변환 수행
        doc_embeddings = self.embeddings.embed_documents(documents)
        
        # 4. ChromaDB에 업서트(Upsert: 기존 ID가 있으면 덮어쓰기)
        collection.upsert(
            ids=ids,
            embeddings=doc_embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)

    def search_top_product(self, query_text: str) -> dict:
        """
        [RAG 1단계: Retrieval]
        유저 소비 특성 텍스트 쿼리를 벡터로 임베딩하여 가장 코사인 유사도가 높은 상위 1개 상품을 반환합니다.
        """
        collection = self.get_or_create_collection()
        
        # 입력 텍스트(유저 재무 상태 요약문)를 쿼리 벡터로 전환
        query_vector = self.embeddings.embed_query(query_text)

        # 유사도 상위 1개 상품(n_results=1) 검색
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )

        # 검색 결과 예외 처리
        if not results or not results["metadatas"] or not results["metadatas"][0]:
            raise ValueError("ChromaDB 지식 DB에서 일치하는 금융상품을 찾지 못했습니다.")

        top_metadata = results["metadatas"][0][0]
        
        # 메타데이터에 문자열로 들어있던 details_json을 파이썬 Dict 객체로 역직렬화
        details_dict = json.loads(top_metadata.get("details_json", "{}"))

        # 서비스 레이어에서 활용하기 좋은 딕셔너리 포맷으로 재구성하여 반환
        return {
            "product_id": top_metadata["product_id"],
            "product_name": top_metadata["product_name"],
            "product_type": top_metadata["product_type"],
            "provider": top_metadata["provider"],
            "summary": top_metadata["summary"],
            "target_group": top_metadata.get("target_group"),
            "njob_trend_tip": top_metadata.get("njob_trend_tip"),
            "details": details_dict
        }

# 전역에서 재사용할 ChromaDB 매니저 싱글톤 인스턴스 생성
chroma_manager = ChromaManager()