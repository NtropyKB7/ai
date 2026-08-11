import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.rag.chroma_client import chroma_manager
from app.schemas.product import (
    CategoryExpenseSummary,
    FinancialProductSchema,
    JobInsightInput,
    ProductRecommendationRequest,
    ProductRecommendationResponse,
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    금융상품 추천과 AI 리포트 문구 생성을 담당합니다.

    핵심 원칙:
    - 원천 거래 전체를 LLM에 보내지 않습니다.
    - 개인정보성 상세 데이터를 LLM에 보내지 않습니다.
    - Java AI-service가 만든 월별 집계 데이터만 사용합니다.
    - 추천 판단에 필요한 값은 Python 로직으로 먼저 정리합니다.
    - LLM은 최종 문구 생성과 설명력 강화에 집중합니다.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )

    async def generate_recommendation(
        self,
        request: ProductRecommendationRequest,
    ) -> ProductRecommendationResponse:
        """
        월별 소득·소비 집계 데이터를 기반으로 추천 상품과 리포트 인사이트를 생성합니다.
        """

        category_expenses = self._parse_category_expenses(request.category_expenses)
        financial_type = self._classify_financial_type(request)

        financial_activity_insight = self._build_financial_activity_insight(
            request=request,
            category_expenses=category_expenses,
            financial_type=financial_type,
        )

        job_insight = self._build_job_insight(request.job_insight_inputs)

        search_query = self._build_product_search_query(
            request=request,
            category_expenses=category_expenses,
            financial_type=financial_type,
        )

        retrieved_product = chroma_manager.search_top_product(query_text=search_query)
        product = FinancialProductSchema(**retrieved_product)

        simulated_extra_income = self._calculate_simulated_extra_income(
            product=retrieved_product,
            request=request,
            category_expenses=category_expenses,
        )

        llm_result = await self._generate_llm_insights(
            request=request,
            product=product,
            simulated_extra_income=simulated_extra_income,
            financial_type=financial_type,
            financial_activity_insight=financial_activity_insight,
            job_insight=job_insight,
            category_expenses=category_expenses,
        )

        return ProductRecommendationResponse(
            recommended_product=product,
            simulated_extra_income=simulated_extra_income,
            reasoning=llm_result.get(
                "reasoning",
                self._fallback_reasoning(product, simulated_extra_income),
            ),
            financial_activity_insight=llm_result.get(
                "financial_activity_insight",
                financial_activity_insight,
            ),
            financial_type=llm_result.get("financial_type", financial_type),
            job_insight=llm_result.get("job_insight", job_insight),
            future_income_trend=llm_result.get(
                "future_income_trend",
                self._fallback_future_income_trend(request),
            ),
        )

    def _parse_category_expenses(
        self,
        category_expenses_text: str,
    ) -> list[CategoryExpenseSummary]:
        """
        Java에서 전달한 category_expenses JSON 문자열을 파싱합니다.

        현재 Java는 List 형태의 JSON 문자열을 전달합니다.
        과거 Map 형태로 넘어오는 경우도 방어적으로 처리합니다.
        """

        if not category_expenses_text:
            return []

        try:
            raw_value = json.loads(category_expenses_text)
        except json.JSONDecodeError:
            logger.warning("category_expenses JSON 파싱 실패: %s", category_expenses_text)
            return []

        if isinstance(raw_value, list):
            return [
                CategoryExpenseSummary(**item)
                for item in raw_value
                if isinstance(item, dict)
            ]

        if isinstance(raw_value, dict):
            return [
                CategoryExpenseSummary(
                    category=category,
                    amount=int(amount or 0),
                    ratio=None,
                )
                for category, amount in raw_value.items()
            ]

        return []

    def _classify_financial_type(
        self,
        request: ProductRecommendationRequest,
    ) -> str:
        """
        사용자의 재무 유형을 설명 가능한 규칙으로 판정합니다.
        """

        total_income = request.total_income or 0
        total_expense = request.total_expense or 0
        available_funds = request.available_funds or 0

        if total_income <= 0:
            return "소득 데이터 부족형"

        available_ratio = available_funds / total_income
        expense_ratio = total_expense / total_income

        if available_funds < 0:
            return "현금흐름 위험형"

        if available_ratio >= 0.3:
            return "저축 여력형"

        if available_ratio >= 0.1:
            return "균형 관리형"

        if expense_ratio >= 0.9:
            return "소비 압박형"

        return "가용자금 관리형"

    def _build_financial_activity_insight(
        self,
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        financial_type: str,
    ) -> str:
        """
        소득·소비 데이터를 기반으로 재무활동 요약 문구를 생성합니다.
        LLM 실패 시 fallback 문구로도 사용됩니다.
        """

        top_category = self._find_top_category(category_expenses)

        if top_category:
            top_category_text = (
                f"가장 큰 소비 카테고리는 "
                f"{top_category.displayName or top_category.category}"
                f"({self._format_won(top_category.amount)})입니다."
            )
        else:
            top_category_text = "카테고리별 소비 데이터는 아직 충분하지 않습니다."

        return (
            f"이번 달 총소득은 {self._format_won(request.total_income)}, "
            f"총소비는 {self._format_won(request.total_expense)}, "
            f"가용자금은 {self._format_won(request.available_funds)}입니다. "
            f"현재 재무 유형은 {financial_type}이며, {top_category_text}"
        )

    def _build_job_insight(
        self,
        job_inputs: list[JobInsightInput],
    ) -> str:
        """
        job_insight_inputs를 기반으로 잡 관련 인사이트를 생성합니다.
        """

        if not job_inputs:
            return "잡별 소득·근무시간 데이터가 충분하지 않아 N잡 인사이트를 생성하기 어렵습니다."

        primary_job = max(
            job_inputs,
            key=lambda job: job.incomeAmount or 0,
        )

        job_name = primary_job.jobName or "주요 잡"
        income_amount = self._format_won(primary_job.incomeAmount or 0)
        work_minutes = primary_job.totalWorkMinutes or 0
        work_hours = round(work_minutes / 60, 1) if work_minutes else 0

        fatigue_text = ""
        if primary_job.averageFatigue is not None:
            fatigue_text = f" 평균 피로도는 {primary_job.averageFatigue}입니다."

        if work_hours > 0:
            return (
                f"{job_name}에서 가장 많은 소득이 발생했습니다. "
                f"월 소득은 {income_amount}, 총 근무시간은 약 {work_hours}시간입니다."
                f"{fatigue_text}"
            )

        return f"{job_name}에서 가장 많은 소득이 발생했습니다. 월 소득은 {income_amount}입니다.{fatigue_text}"

    def _build_product_search_query(
        self,
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        financial_type: str,
    ) -> str:
        """
        ChromaDB 검색에 사용할 쿼리 문장을 만듭니다.

        RAG는 최종 판단자가 아니라 후보 상품 검색기로 사용합니다.
        """

        top_category = self._find_top_category(category_expenses)

        query_parts = [
            f"재무유형: {financial_type}",
            f"가용자금: {request.available_funds}원",
            f"총소득: {request.total_income}원",
            f"총소비: {request.total_expense}원",
        ]

        if top_category:
            query_parts.append(f"주요 소비 카테고리: {top_category.category}")

        if request.income_change_rate is not None:
            query_parts.append(f"전월 대비 소득 증감률: {request.income_change_rate}")

        if request.income_volatility is not None:
            query_parts.append(f"소득 변동성: {request.income_volatility}")

        return " | ".join(query_parts)

    def _calculate_simulated_extra_income(
        self,
        product: dict[str, Any],
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
    ) -> int:
        """
        추천 상품 사용 시 월 단위 예상 추가 수익 또는 절감액을 계산합니다.
        """

        product_type = product.get("product_type")
        details = product.get("details", {})

        if product_type == "SAVINGS":
            interest_rate = float(details.get("interest_rate", 3.0)) / 100
            available_funds = max(request.available_funds or 0, 0)
            return int((available_funds * interest_rate) / 12)

        if product_type == "CARD":
            discount_rate = float(details.get("discount_rate", 0.1))
            top_category = self._find_top_category(category_expenses)

            base_expense = top_category.amount if top_category else request.total_expense
            return int(max(base_expense or 0, 0) * discount_rate)

        return 0

    async def _generate_llm_insights(
        self,
        request: ProductRecommendationRequest,
        product: FinancialProductSchema,
        simulated_extra_income: int,
        financial_type: str,
        financial_activity_insight: str,
        job_insight: str,
        category_expenses: list[CategoryExpenseSummary],
    ) -> dict[str, str]:
        """
        LLM을 호출해 최종 리포트 문구를 생성합니다.
        LLM에는 집계 데이터와 요약된 판단 결과만 전달합니다.
        """

        category_summary = [
            {
                "category": item.category,
                "displayName": item.displayName,
                "amount": item.amount,
                "ratio": item.ratio,
            }
            for item in category_expenses
        ]

        job_summary = [
            {
                "jobId": item.jobId,
                "jobName": item.jobName,
                "incomeAmount": item.incomeAmount,
                "incomeRatio": item.incomeRatio,
                "totalWorkMinutes": item.totalWorkMinutes,
                "workDays": item.workDays,
                "averageFatigue": item.averageFatigue,
                "latestFatigue": item.latestFatigue,
            }
            for item in request.job_insight_inputs
        ]

        prompt = f"""
당신은 N잡러를 위한 금융상품 추천 및 월간 재무 리포트 작성 전문가입니다.

아래 데이터는 원천 거래가 아니라 월별 집계 데이터입니다.
개인정보나 원천 거래 상세는 포함되어 있지 않습니다.

[사용자 월간 재무 집계]
- 기준 연월: {request.year_month}
- 총소득: {request.total_income}원
- 총소비: {request.total_expense}원
- 가용자금: {request.available_funds}원
- 전월 대비 소득 증감액: {request.income_change_amount}
- 전월 대비 소득 증감률: {request.income_change_rate}
- 소득 변동성: {request.income_volatility}
- 카테고리별 소비: {json.dumps(category_summary, ensure_ascii=False)}
- 잡별 인사이트 입력: {json.dumps(job_summary, ensure_ascii=False)}

[사전 계산된 판단]
- 재무 유형: {financial_type}
- 재무활동 요약: {financial_activity_insight}
- 잡 관련 요약: {job_insight}

[추천 금융상품]
- 상품명: {product.product_name}
- 제공사: {product.provider}
- 상품 유형: {product.product_type}
- 상품 요약: {product.summary}
- N잡 활용 팁: {product.njob_trend_tip}
- 예상 추가 수익 또는 절감액: {simulated_extra_income}원

[작성 규칙]
1. reasoning은 추천 상품을 쓰면 어떤 이득이 있는지 직관적으로 설명하세요.
2. financial_activity_insight는 소비와 소득 흐름을 자연스럽게 요약하세요.
3. financial_type은 주어진 재무 유형을 유지하되, 더 적절한 표현으로 다듬어도 됩니다.
4. job_insight는 잡별 소득, 근무시간, 피로도 관점의 조언으로 작성하세요.
5. future_income_trend는 다음 달 소득 흐름과 N잡 코칭 문구로 작성하세요.
6. 과장된 수익 보장 표현은 쓰지 마세요.
7. 반드시 JSON 객체만 반환하세요.

[응답 형식]
{{
  "reasoning": "...",
  "financial_activity_insight": "...",
  "financial_type": "...",
  "job_insight": "...",
  "future_income_trend": "..."
}}
"""

        try:
            response = await self.llm.ainvoke(prompt)
            return self._parse_llm_json(response.content)

        except Exception as error:
            logger.error("추천 LLM 문구 생성 실패: %s", str(error), exc_info=True)
            return {}

    def _parse_llm_json(
        self,
        content: str,
    ) -> dict[str, str]:
        """
        LLM 응답을 JSON으로 파싱합니다.
        실패하면 빈 dict를 반환하고 fallback 문구를 사용합니다.
        """

        if not content:
            return {}

        cleaned = content.strip()
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("LLM JSON 파싱 실패: %s", content)
            return {}

        if not isinstance(parsed, dict):
            return {}

        return {
            key: str(value)
            for key, value in parsed.items()
            if value is not None
        }

    def _find_top_category(
        self,
        category_expenses: list[CategoryExpenseSummary],
    ) -> CategoryExpenseSummary | None:
        """
        소비 금액이 가장 큰 카테고리를 찾습니다.
        """

        if not category_expenses:
            return None

        return max(category_expenses, key=lambda item: item.amount or 0)

    def _fallback_reasoning(
        self,
        product: FinancialProductSchema,
        simulated_extra_income: int,
    ) -> str:
        """
        LLM 실패 시 사용할 추천 사유 기본 문구입니다.
        """

        return (
            f"{product.product_name}은 현재 소비·소득 흐름에 맞춰 활용할 수 있는 상품입니다. "
            f"예상 추가 수익 또는 절감액은 약 {self._format_won(simulated_extra_income)}입니다."
        )

    def _fallback_future_income_trend(
        self,
        request: ProductRecommendationRequest,
    ) -> str:
        """
        LLM 실패 시 사용할 미래 소득 트렌드 기본 문구입니다.
        """

        if request.income_change_rate is None:
            return "소득 변화 데이터가 충분하지 않아 다음 달 흐름은 추가 관찰이 필요합니다."

        if request.income_change_rate > 0:
            return "최근 소득이 증가하는 흐름이므로, 가용자금을 저축이나 혜택형 상품에 나누어 활용할 여지가 있습니다."

        if request.income_change_rate < 0:
            return "최근 소득이 감소하는 흐름이므로, 소비 조정과 현금흐름 안정이 우선입니다."

        return "최근 소득은 큰 변동 없이 유지되고 있어 안정적인 관리가 가능합니다."

    def _format_won(
        self,
        amount: int | None,
    ) -> str:
        """
        원 단위 금액을 읽기 쉬운 문자열로 변환합니다.
        """

        return f"{amount or 0:,}원"


recommendation_service = RecommendationService()