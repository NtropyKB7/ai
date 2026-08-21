import json
import logging
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

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
from app.services.financial_product_interest_service import FinancialProductInterestService
from app.services.personalized_product_scoring_service import (
    CandidateScore,
    PersonalizedProductScoringService,
)
from app.services.recommendation_profile_service import RecommendationProfileService

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    금융상품 추천과 AI 리포트 문구 생성을 담당하는 서비스입니다.

    이번 리팩토링의 핵심 목표는 두 가지입니다.
    1. 추천 상품 선택 로직과 "문구 생성 로직"의 책임을 더 명확히 분리한다.
    2. LLM 실패 시에도 응답 문구 품질이 크게 떨어지지 않도록 fallback을 보강한다.

    설계 원칙:
    - 원천 거래 전체를 LLM에 전달하지 않습니다.
    - 개인정보성 세부 데이터는 LLM에 전달하지 않습니다.
    - Java AI-service가 전달한 월별 집계 데이터만 사용합니다.
    - RAG는 후보 검색용으로만 사용하고, 최종 추천 선택은 Python 규칙이 보완합니다.
    """

    def __init__(self, llm: ChatOpenAI | None = None, chroma=None):
        # 테스트에서는 외부 LLM 대신 mock 객체를 넣을 수 있도록 주입 구조를 열어둡니다.
        self.llm = llm or ChatOpenAI(
            model=settings.OPENAI_GENERATION_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )

        # 검색 매니저도 주입 가능하게 두어, 테스트에서 외부 의존성을 줄일 수 있게 합니다.
        self.chroma_manager = chroma or chroma_manager
        self.profile_service = RecommendationProfileService()
        self.scoring_service = PersonalizedProductScoringService()
        self.interest_service = FinancialProductInterestService()

    async def generate_recommendation(
        self,
        request: ProductRecommendationRequest,
    ) -> ProductRecommendationResponse:
        """
        월별 집계 데이터를 바탕으로 최종 추천 응답을 생성합니다.

        흐름:
        1. 입력 집계 데이터 해석
        2. 재무 유형/소비 패턴/잡 인사이트 계산
        3. RAG 후보 검색
        4. Python 규칙 점수화로 최종 추천 상품 선택
        5. LLM으로 문구 품질 보강
        6. LLM 실패 시 fallback 문구 사용
        """
        category_expenses = self._parse_category_expenses(request.category_expenses)
        profile = self.profile_service.build(request, category_expenses)
        if profile.investment_budget <= 0:
            raise ValueError("추천 가능한 투자 예산이 없습니다.")
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
        search_query = f"{search_query} | {profile.search_document()}"

        if hasattr(self.chroma_manager, "search_finlife_products_with_scores"):
            candidate_products, similarities = (
                self.chroma_manager.search_finlife_products_with_scores(search_query)
            )
        else:
            candidate_products = self.chroma_manager.search_products(
                query_text=search_query, n_results=10_000
            )
            similarities = {}
        ranked = self.scoring_service.score_all(
            candidate_products, profile, similarities
        )
        if not ranked:
            raise ValueError("추천 후보 상품이 비어 있습니다.")
        selected = ranked[0]
        selected_product = self._recommendation_product(selected)

        product = FinancialProductSchema(**selected_product)

        simulated_extra_income = self._calculate_simulated_extra_income(
            product=selected_product,
            request=request,
            category_expenses=category_expenses,
            profile=profile,
        )

        # 각 응답 필드의 기본값을 먼저 만들어 둡니다.
        # 이렇게 하면 LLM이 일부 필드만 생성해도 나머지는 안정적으로 채울 수 있습니다.
        default_texts = self._build_default_response_texts(
            request=request,
            product=product,
            simulated_extra_income=simulated_extra_income,
            financial_type=financial_type,
            financial_activity_insight=financial_activity_insight,
            job_insight=job_insight,
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

        # Product reasoning is evidence-bound and deterministic. LLM prose is
        # accepted only for the remaining report fields.
        merged_texts = self._merge_llm_texts_with_defaults(
            llm_result={k: v for k, v in llm_result.items() if k != "reasoning"},
            default_texts=default_texts,
        )

        return ProductRecommendationResponse(
            recommended_product=product,
            simulated_extra_income=simulated_extra_income,
            reasoning=merged_texts["reasoning"],
            financial_activity_insight=merged_texts["financial_activity_insight"],
            financial_type=merged_texts["financial_type"],
            job_insight=merged_texts["job_insight"],
            future_income_trend=merged_texts["future_income_trend"],
        )

    def _parse_category_expenses(
        self,
        category_expenses_text: str,
    ) -> list[CategoryExpenseSummary]:
        """
        Java에서 전달한 category_expenses JSON 문자열을 파싱합니다.

        현재는 List 형태 JSON 문자열을 기본으로 기대하지만,
        이전 호환을 위해 Map 구조도 방어적으로 허용합니다.
        """
        if not category_expenses_text:
            return []

        try:
            raw_value = json.loads(category_expenses_text)
        except json.JSONDecodeError:
            logger.warning("category_expenses JSON 파싱 실패")
            return []

        if isinstance(raw_value, list):
            parsed = []
            for item in raw_value:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed.append(CategoryExpenseSummary(**item))
                except ValidationError:
                    continue
            return parsed

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
        사용자의 월간 재무 상태를 간단한 규칙으로 분류합니다.

        이 값은:
        - 최종 추천 상품 선택 점수화
        - 응답용 financial_type
        - 문구 생성의 기준 맥락
        에 모두 사용됩니다.
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
        소비·소득 흐름을 설명하는 기본 문구를 생성합니다.

        역할:
        - "이번 달 재무 활동이 전반적으로 어떤 상태인가"를 요약
        - 특정 상품 추천 사유(reasoning)와는 분리
        """
        top_category = self._find_top_category(category_expenses)
        cash_flow_text = self._describe_cash_flow(request)

        if top_category:
            category_label = top_category.displayName or top_category.category
            category_text = (
                f"가장 큰 소비 카테고리는 {category_label}이며, "
                f"지출 규모는 {self._format_won(top_category.amount)}입니다."
            )
        else:
            category_text = "카테고리별 소비 데이터는 아직 충분하지 않습니다."

        return (
            f"{cash_flow_text} 현재 재무 상태는 {financial_type}에 가깝고, "
            f"{category_text}"
        )

    def _build_job_insight(
        self,
        job_inputs: list[JobInsightInput],
    ) -> str:
        """
        잡별 소득·근무시간·피로도 데이터를 요약합니다.

        역할:
        - 재무 인사이트와 분리해서 "일/수익 구조"만 요약
        - 미래 소득 전망 문구와도 겹치지 않도록 현재 상태 중심으로 작성
        """
        if not job_inputs:
            return "잡별 소득·근무시간 데이터가 충분하지 않아 N잡 인사이트를 생성하기 어렵습니다."

        primary_job = max(job_inputs, key=lambda job: job.incomeAmount or 0)

        job_name = primary_job.jobName or "주요 잡"
        income_amount = self._format_won(primary_job.incomeAmount or 0)
        work_minutes = primary_job.totalWorkMinutes or 0
        work_hours = round(work_minutes / 60, 1) if work_minutes else 0

        fatigue_sentence = ""
        if primary_job.averageFatigue is not None:
            fatigue_sentence = f" 평균 피로도는 {primary_job.averageFatigue} 수준입니다."

        if work_hours > 0:
            return (
                f"{job_name}에서 가장 큰 소득 비중이 발생했습니다. "
                f"총소득은 {income_amount}, 총 근무시간은 약 {work_hours}시간입니다."
                f"{fatigue_sentence}"
            )

        return (
            f"{job_name}에서 가장 큰 소득 비중이 발생했습니다. "
            f"총소득은 {income_amount}입니다.{fatigue_sentence}"
        )

    def _build_product_search_query(
        self,
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        financial_type: str,
    ) -> str:
        """
        ChromaDB 검색용 요약 쿼리를 생성합니다.

        주의:
        - 프롬프트처럼 길게 만들지 않고, 검색에 필요한 핵심 정보만 담습니다.
        - top category를 1개만 쓰지 않고 상위 2개까지 넣어 검색 편향을 줄입니다.
        """
        top_categories = sorted(
            category_expenses,
            key=lambda item: item.amount or 0,
            reverse=True,
        )[:2]

        query_parts = [
            f"재무유형: {financial_type}",
            f"가용자금 {request.available_funds}원",
            f"총소득 {request.total_income}원",
            f"총소비 {request.total_expense}원",
        ]

        for category in top_categories:
            query_parts.append(f"주요 소비 카테고리: {category.category}")

        if request.income_change_rate is not None:
            query_parts.append(f"전월 대비 소득 증감률: {request.income_change_rate}")

        if request.income_volatility is not None:
            query_parts.append(f"소득 변동성: {request.income_volatility}")

        return " | ".join(query_parts)

    def _select_best_product(
        self,
        products: list[dict[str, Any]],
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        financial_type: str,
    ) -> dict[str, Any]:
        """
        검색된 상품 후보 중 현재 사용자 상황에 가장 잘 맞는 상품을 선택합니다.
        """
        if not products:
            raise ValueError("추천 후보 상품이 비어 있습니다.")

        scored_products = [
            (
                self._score_product_candidate(
                    product=product,
                    request=request,
                    category_expenses=category_expenses,
                    financial_type=financial_type,
                ),
                product,
            )
            for product in products
        ]

        scored_products.sort(key=lambda item: item[0], reverse=True)
        return scored_products[0][1]

    def _score_product_candidate(
        self,
        product: dict[str, Any],
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        financial_type: str,
    ) -> float:
        """
        후보 상품 하나의 적합도 점수를 계산합니다.

        큰 방향:
        - 저축 여력이 있으면 SAVINGS 쪽 가중치
        - 소비 압박이 크면 CARD 쪽 가중치
        - 카드면 소비 카테고리 혜택 매칭을 반영
        - 적금이면 금리와 가용자금/변동성을 반영
        """
        score = 0.0
        product_type = product.get("product_type")
        details = product.get("details", {})

        top_category = self._find_top_category(category_expenses)
        available_funds = request.available_funds or 0
        income_change_rate = request.income_change_rate or 0.0
        income_volatility = request.income_volatility or 0.0

        if financial_type in {"저축 여력형", "균형 관리형"} and product_type == "SAVINGS":
            score += 4.0

        if financial_type in {"소비 압박형", "현금흐름 위험형"} and product_type == "CARD":
            score += 4.0

        if available_funds > 500_000 and product_type == "SAVINGS":
            score += 2.0

        if available_funds <= 300_000 and product_type == "CARD":
            score += 2.0

        if income_change_rate > 0 and product_type == "SAVINGS":
            score += 1.0

        if income_volatility >= 0.2 and product_type == "SAVINGS":
            score += 0.5

        if product_type == "CARD" and top_category:
            matched_categories = self._extract_discount_categories(product)

            if top_category.category in matched_categories:
                score += 4.0
            elif top_category.displayName and top_category.displayName in product.get("summary", ""):
                score += 2.0

            discount_rate = float(details.get("discount_rate", 0.0))
            score += discount_rate * 10

        if product_type == "SAVINGS":
            interest_rate = float(details.get("interest_rate", 0.0))
            score += interest_rate

        return score

    def _extract_discount_categories(
        self,
        product: dict[str, Any],
    ) -> set[str]:
        """
        카드 상품의 혜택 카테고리 코드를 추출합니다.
        """
        details = product.get("details", {})
        raw_categories = details.get("discount_categories", [])

        if not isinstance(raw_categories, list):
            raw_categories = []

        return {str(category).upper() for category in raw_categories}

    def _calculate_simulated_extra_income(
        self,
        product: dict[str, Any],
        request: ProductRecommendationRequest,
        category_expenses: list[CategoryExpenseSummary],
        profile=None,
    ) -> int:
        """
        추천 상품을 사용했을 때의 예상 추가 수익 또는 절감 금액을 계산합니다.
        """
        product_type = product.get("product_type")
        details = product.get("details", {})

        if product_type in {"SAVINGS", "DEPOSIT"}:
            effective_profile = profile or self.profile_service.build(
                request, category_expenses
            )
            return self.interest_service.monthly_interest(
                product_type=product_type,
                investment_budget=effective_profile.investment_budget,
                annual_base_rate=Decimal(str(details["interest_rate"])),
                term_months=int(details["term_months"]),
            )

        if product_type == "CARD":
            discount_rate = float(details.get("discount_rate", 0.1))
            max_monthly_benefit = int(
                details.get("maxMonthlyBenefit")
                or details.get("max_monthly_benefit")
                or 0
            )

            matching_category = self._find_best_matching_category(
                product=product,
                category_expenses=category_expenses,
            )

            if matching_category is not None:
                eligible_expense = matching_category.amount or 0
            else:
                top_category = self._find_top_category(category_expenses)
                eligible_expense = (top_category.amount if top_category else request.total_expense) or 0

            simulated_benefit = int(max(eligible_expense, 0) * discount_rate)

            if max_monthly_benefit > 0:
                simulated_benefit = min(simulated_benefit, max_monthly_benefit)

            return simulated_benefit

        return 0

    @staticmethod
    def _recommendation_product(scored: CandidateScore) -> dict[str, Any]:
        product = dict(scored.product)
        details = dict(product.get("details") or {})
        details["recommendation_score_breakdown"] = {
            "financial_benefit_score": scored.financial_benefit_score,
            "semantic_relevance_score": scored.semantic_relevance_score,
            "financial_suitability_score": scored.financial_suitability_score,
            "condition_match_score": scored.condition_match_score,
            "personalized_score": scored.personalized_score,
        }
        details["recommendation_evidence"] = [
            {
                "evidence_type": item.evidence_type.value,
                "source_field": item.source_field,
                "normalized_key": item.normalized_key,
            }
            for item in scored.evidences
        ]
        product["details"] = details
        return product

    def _find_best_matching_category(
        self,
        product: dict[str, Any],
        category_expenses: list[CategoryExpenseSummary],
    ) -> CategoryExpenseSummary | None:
        """
        카드 혜택 카테고리와 가장 잘 맞는 소비 카테고리를 찾습니다.
        """
        discount_categories = self._extract_discount_categories(product)
        if not discount_categories:
            return None

        matched_categories = [
            item
            for item in category_expenses
            if item.category.upper() in discount_categories
        ]

        if not matched_categories:
            return None

        return max(matched_categories, key=lambda item: item.amount or 0)

    def _build_default_response_texts(
        self,
        request: ProductRecommendationRequest,
        product: FinancialProductSchema,
        simulated_extra_income: int,
        financial_type: str,
        financial_activity_insight: str,
        job_insight: str,
    ) -> dict[str, str]:
        """
        각 응답 필드의 기본 문구를 생성합니다.

        이 메서드를 별도로 둔 이유:
        - 필드별 책임을 명확하게 나누기 위해
        - LLM이 일부 필드를 누락해도 전체 응답 품질을 유지하기 위해
        """
        return {
            "reasoning": self._fallback_reasoning(
                product=product,
                simulated_extra_income=simulated_extra_income,
                request=request,
            ),
            "financial_activity_insight": financial_activity_insight,
            "financial_type": financial_type,
            "job_insight": job_insight,
            "future_income_trend": self._fallback_future_income_trend(request),
        }

    def _merge_llm_texts_with_defaults(
        self,
        llm_result: dict[str, str],
        default_texts: dict[str, str],
    ) -> dict[str, str]:
        """
        LLM 결과와 fallback 기본값을 병합합니다.

        규칙:
        - LLM 값이 비어 있거나 너무 짧으면 기본값 유지
        - financial_type은 과도하게 장황해지지 않도록 정리
        """
        merged = dict(default_texts)

        for key, default_value in default_texts.items():
            llm_value = self._normalize_text(llm_result.get(key))

            if not llm_value:
                continue

            # 너무 짧은 문구는 품질이 낮다고 보고 기본값을 유지합니다.
            # 단 financial_type은 라벨형 필드라 예외적으로 짧아도 허용합니다.
            if key != "financial_type" and len(llm_value) < 8:
                continue

            merged[key] = llm_value

        merged["financial_type"] = self._normalize_financial_type_label(
            merged["financial_type"]
        )
        return merged

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
        LLM으로 최종 리포트 문구를 생성합니다.

        여기서는 이미 계산된 요약값만 전달합니다.
        즉, LLM은 "판단기"라기보다 "설명 문구 보강기" 역할에 가깝습니다.
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
        allowed_evidence = product.details.get("recommendation_evidence") or []
        score_breakdown = product.details.get("recommendation_score_breakdown") or {}
        selected_option = product.details.get("selected_option") or {}

        # 프롬프트도 필드별 역할을 더 분명히 적어 주어,
        # reasoning / financial_activity_insight / job_insight / future_income_trend가
        # 서로 비슷한 말만 반복하지 않도록 유도합니다.
        prompt = f"""
당신은 N잡러를 위한 금융상품 추천 및 월간 재무 리포트 작성 전문가입니다.

아래 데이터는 원천 거래가 아니라 월별 집계 데이터입니다.
개인정보나 거래 상세 내역은 포함되어 있지 않습니다.

[사용자 월간 재무 집계]
- 기준 연월: {request.year_month}
- 총소득: {request.total_income}원
- 총소비: {request.total_expense}원
- 가용자금: {request.available_funds}원
- 전월 대비 소득 증감액: {request.income_change_amount}
- 전월 대비 소득 증감률: {request.income_change_rate}
- 소득 변동성: {request.income_volatility}
- 카테고리별 소비: {json.dumps(category_summary, ensure_ascii=False)}
- 잡 인사이트 입력: {json.dumps(job_summary, ensure_ascii=False)}

[사전 계산된 기본 문구]
- 재무 유형: {financial_type}
- 재무활동 요약: {financial_activity_insight}
- 잡 관련 요약: {job_insight}

[추천 금융상품]
- 상품명: {product.product_name}
- 제공사: {product.provider}
- 상품 유형: {product.product_type}
- 상품 요약: {product.summary}
- N잡 활용 팁: {product.njob_trend_tip}
- 선택 옵션 기본금리: {selected_option.get('base_interest_rate')}
- 선택 옵션 최고우대금리(계산 미사용): {selected_option.get('preferred_interest_rate')}
- 선택 옵션 기간(개월): {selected_option.get('term_months')}
- 허용된 개인화 근거: {json.dumps(allowed_evidence, ensure_ascii=False)}
- 점수 breakdown: {json.dumps(score_breakdown, ensure_ascii=False)}
- 예상 추가 수익 또는 절감액: {simulated_extra_income}원

[필드별 작성 책임]
1. reasoning
   - 왜 이 상품이 현재 사용자 상황에 맞는지 설명
   - 상품 특징 + 사용자 재무 상태 + 기대 효과를 자연스럽게 연결
2. financial_activity_insight
   - 이번 달 소비/소득 흐름 자체를 요약
   - 상품 추천 이유와 같은 말 반복 금지
3. financial_type
   - 사용자의 재무 상태를 짧은 라벨로 표현
4. job_insight
   - 잡별 소득, 근무시간, 피로도 관점의 현재 상태 요약
5. future_income_trend
   - 미래를 과장 예측하지 말고, 현재 흐름 기준의 다음 달 관리 조언 제공

[공통 규칙]
- 과장된 수익 보장 표현은 금지합니다.
- 허용된 개인화 근거에 없는 혜택, 대상, 우대조건은 생성하지 않습니다.
- 의미 유사도 점수만으로 개인화 사실을 단정하지 않습니다.
- 각 필드는 서로 다른 역할을 가져야 합니다.
- 수치 나열만 하지 말고 해석형 문장으로 작성합니다.
- 반드시 JSON 객체만 반환합니다.

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
        except Exception:
            # 하위 OpenAI 예외 및 요청 원문을 외부 로그에 노출하지 않습니다.
            logger.error("추천 LLM 문구 생성 실패")
            return {}

    def _parse_llm_json(
        self,
        content: str,
    ) -> dict[str, str]:
        """
        LLM 응답을 JSON으로 파싱합니다.
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
            logger.warning("LLM JSON 파싱 실패")
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
        소비 금액이 가장 큰 카테고리를 반환합니다.
        """
        if not category_expenses:
            return None

        return max(category_expenses, key=lambda item: item.amount or 0)

    def _fallback_reasoning(
        self,
        product: FinancialProductSchema,
        simulated_extra_income: int,
        request: ProductRecommendationRequest,
    ) -> str:
        """
        LLM 실패 시 사용할 reasoning 기본 문구입니다.

        reasoning은 "상품 추천 이유" 전용 필드이므로,
        financial_activity_insight나 future_income_trend와 겹치지 않게
        상품 중심으로만 작성합니다.
        """
        if product.product_type == "SAVINGS":
            evidence = product.details.get("recommendation_evidence") or []
            structure = ""
            kinds = {item.get("evidence_type") for item in evidence}
            if "FLEXIBLE_INSTALLMENT_MATCH" in kinds:
                structure = " 소득 흐름 변화에 대응할 수 있는 자유적립 구조가 확인됐습니다."
            elif "STABLE_FIXED_INSTALLMENT_MATCH" in kinds:
                structure = " 현재 현금흐름과 맞는 정액적립 구조가 확인됐습니다."
            return (
                f"{product.product_name}은 선택된 보장 기본금리와 가입 기간을 기준으로 비교된 적금입니다."
                f"{structure} 이번 달 잉여자금의 50%를 월 납입액으로 가정한 월 예상 세전 이자는 "
                f"약 {self._format_won(simulated_extra_income)}입니다."
            )

        if product.product_type == "DEPOSIT":
            return (
                f"{product.product_name}은 선택된 보장 기본금리와 가입 기간을 기준으로 비교된 정기예금입니다. "
                "예치 원금은 전체 보유 목돈이 아니라 이번 달 잉여자금의 50%만 일시 예치한다고 가정했습니다. "
                f"월 예상 세전 이자는 약 {self._format_won(simulated_extra_income)}입니다."
            )

        return (
            f"{product.product_name}은 현재 소비 흐름에서 직접적인 절감 효과를 기대할 수 있는 카드 상품입니다. "
            f"예상 절감 금액은 약 {self._format_won(simulated_extra_income)}입니다."
        )

    def _fallback_future_income_trend(
        self,
        request: ProductRecommendationRequest,
    ) -> str:
        """
        LLM 실패 시 사용할 future_income_trend 기본 문구입니다.

        미래를 단정적으로 예측하지 않고,
        "현재 흐름 기준 조언" 형태로 작성합니다.
        """
        if request.income_change_rate is None:
            return "소득 변화 데이터가 아직 충분하지 않아 다음 달 흐름은 추가 관찰이 필요합니다."

        if request.income_change_rate > 0:
            return "최근 소득이 증가하는 흐름이라면 다음 달에도 가용자금 일부를 저축이나 혜택형 상품에 연결해볼 수 있습니다."

        if request.income_change_rate < 0:
            return "최근 소득이 감소하는 흐름이라면 다음 달에는 생활비 절감과 현금흐름 안정화에 더 집중하는 것이 좋습니다."

        return "최근 소득 흐름이 큰 변동 없이 유지되고 있어 안정적인 자금 관리 전략이 적합합니다."

    def _describe_cash_flow(
        self,
        request: ProductRecommendationRequest,
    ) -> str:
        """
        총소득/총소비/가용자금을 한 문장으로 읽기 좋게 요약합니다.
        """
        return (
            f"이번 달 총소득은 {self._format_won(request.total_income)}, "
            f"총소비는 {self._format_won(request.total_expense)}, "
            f"가용자금은 {self._format_won(request.available_funds)}입니다."
        )

    def _normalize_text(
        self,
        text: str | None,
    ) -> str:
        """
        LLM 결과 문자열의 앞뒤 공백을 정리합니다.
        """
        if text is None:
            return ""

        return str(text).strip()

    def _normalize_financial_type_label(
        self,
        text: str,
    ) -> str:
        """
        financial_type 라벨이 지나치게 길어지거나 문장처럼 바뀌는 것을 조금 정리합니다.
        """
        normalized = self._normalize_text(text)
        if not normalized:
            return "재무 유형 분석 중"

        # 너무 길게 생성되면 앞부분만 사용하지 않고 기본적으로 그대로 두되,
        # 문장형 표현일 때는 마침표를 제거해 라벨처럼 보이게 만듭니다.
        return normalized.replace(".", "").strip()

    def _format_won(
        self,
        amount: int | None,
    ) -> str:
        """
        원 단위 금액을 읽기 쉬운 문자열로 변환합니다.
        """
        return f"{amount or 0:,}원"


recommendation_service = RecommendationService()
