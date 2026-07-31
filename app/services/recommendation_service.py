import json
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.schemas.product import (
    FinancialProductSchema, 
    ProductRecommendationRequest, 
    ProductRecommendationResponse
)
from app.rag.chroma_client import chroma_manager

class RecommendationService:
    """
    RAG 파이프라인(Retrieval -> Simulation -> Augment & Generation)을 통해
    유저 맞춤형 금융상품 추천 및 N잡 코칭 리포트를 생성하는 핵심 서비스 클래스.
    """
    def __init__(self):
        # LangChain ChatOpenAI 기반 gpt-4o-mini 모델 초기화
        # temperature를 0.2로 설정하여 환각(Hallucination)을 최소화하고 정교한 문구 유도
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )

    def _calculate_simulation_income(self, product: dict, request: ProductRecommendationRequest) -> int:
        """
        [시뮬레이션 금액 연산 로직]
        유저가 이 상품을 지출/적립에 활용했을 때 얻을 수 있었던 월 예상 이자 수익 또는 절감 금액을 계산합니다.
        """
        p_type = product.get("product_type")
        details = product.get("details", {})

        # 예·적금 상품인 경우: 가용 자금을 연 우대금리로 적립 시 1개월 예상 이자 수익 연산
        if p_type == "SAVINGS":
            interest_rate = details.get("interest_rate", 3.0) / 100 # 연 금리 (예: 4.5% -> 0.045)
            available_funds = max(request.available_funds, 0)
            # 월 이자 계산 (단리 기준): (가용자금 * 연금리) / 12개월
            return int((available_funds * interest_rate) / 12)

        # 카드 상품인 경우: 당월 주유/이동 지출액 대비 할인율 적용 절감액 연산
        elif p_type == "CARD":
            discount_rate = details.get("discount_rate", 0.10) # 카드 할인율 (예: 10% -> 0.1)
            fuel_expense = request.monthly_fuel_expense or 300000 # 지출 미입력 시 기본값 30만원 적용
            return int(fuel_expense * discount_rate)

        return 0

    async def generate_recommendation(self, request: ProductRecommendationRequest) -> ProductRecommendationResponse:
        """
        RAG 파이프라인 3단계를 수행하여 최종 AI 코칭 추천 응답 객체를 반환하는 비동기 메서드.
        """
        # =========================================================================
        # [1단계: Retrieval (지식 검색)]
        # 유저의 소비 패턴 요약문(consumption_summary)을 Vector DB 쿼리로 날려 최적 상품 1개 검색
        # =========================================================================
        retrieved_dict = chroma_manager.search_top_product(query_text=request.consumption_summary)
        product_dto = FinancialProductSchema(**retrieved_dict)

        # =========================================================================
        # [2단계: Simulation (수익/절감액 산출)]
        # 유저의 가용 자금/주유 지출액 데이터를 바탕으로 미사용 시 손실/사용 시 이득 시뮬레이션 계산
        # =========================================================================
        extra_income = self._calculate_simulation_income(retrieved_dict, request)

        # =========================================================================
        # [3단계: Augment & Generation (지식 주입 및 LLM 추론)]
        # 검색된 지식 + 시뮬레이션 결과 수치를 Prompt Context로 주입하여 정밀한 문구 작성
        # =========================================================================
        prompt = f"""
        당신은 N잡러 전문 금융 파이낸셜 코치입니다. 
        아래 제공된 [유저 재무 스냅샷]과 RAG 지식 DB에서 검증하여 끌어온 [추천 금융상품 지식]만을 기반으로 유저 맞춤형 코칭 리포트 문구를 작성하세요.

        [유저 재무 스냅샷]
        - 당월 가용 자금: {request.available_funds:,}원
        - 주유/이동 지출액: {request.monthly_fuel_expense:,}원
        - 소비 패턴 요약: {request.consumption_summary}

        [추천 금융상품 지식 (RAG 검색 및 시뮬레이션 결과)]
        - 상품명: {product_dto.product_name} ({product_dto.provider})
        - 상품 유형: {product_dto.product_type}
        - 핵심 혜택 요약: {product_dto.summary}
        - N잡 활용 팁: {product_dto.njob_trend_tip}
        - 월 시뮬레이션 추가 수익/절감액: {extra_income:,}원

        [작성 지침]
        1. `reasoning`: "지난달 이 상품을 사용하셨다면 약 {extra_income:,}원의 추가 이자/지출 절감 효과를 보실 수 있었습니다" 맥락을 반드시 포함하여 직관적인 추천 사유 2~3문장 작성.
        2. `future_income_trend`: 유저의 소비/부업 특성과 이 상품을 결합하여 향후 N잡 수익성을 높일 수 있는 코칭 문구 2문장 작성.

        반드시 아래 지정된 JSON 형식으로만 응답을 출력하세요:
        {{
            "reasoning": "...",
            "future_income_trend": "..."
        }}
        """

        # 비동기로 LLM 호출하여 응답 수신
        response = await self.llm.ainvoke(prompt)
        
        # LLM 응답 텍스트 파싱 (마크다운 ```json 태그 제거 후 JSON 객체 변환)
        parsed_res = json.loads(response.content.strip().replace("```json", "").replace("```", ""))

        # Pydantic 응답 DTO로 포장하여 반환
        return ProductRecommendationResponse(
            recommended_product=product_dto,
            simulated_extra_income=extra_income,
            reasoning=parsed_res["reasoning"],
            future_income_trend=parsed_res["future_income_trend"]
        )

# 전역에서 재사용할 추천 서비스 싱글톤 인스턴스 생성
recommendation_service = RecommendationService()