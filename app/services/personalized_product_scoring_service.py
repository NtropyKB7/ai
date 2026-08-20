from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from app.schemas.transaction import ExpenseCategory
from app.services.recommendation_profile_service import RecommendationUserProfile


@dataclass(frozen=True)
class PersonalizedScorePolicy:
    financial_benefit_weight: Decimal = Decimal("0.35")
    semantic_relevance_weight: Decimal = Decimal("0.20")
    financial_suitability_weight: Decimal = Decimal("0.30")
    condition_match_weight: Decimal = Decimal("0.15")
    condition_evidence_points: Decimal = Decimal("0.20")

    def __post_init__(self):
        total = (
            self.financial_benefit_weight
            + self.semantic_relevance_weight
            + self.financial_suitability_weight
            + self.condition_match_weight
        )
        if total != Decimal("1"):
            raise ValueError("Personalized score weights must add up to 1")


class EvidenceType(StrEnum):
    MULTI_INCOME_MATCH = "MULTI_INCOME_MATCH"
    SALARY_CONDITION_MATCH = "SALARY_CONDITION_MATCH"
    BUSINESS_INCOME_MATCH = "BUSINESS_INCOME_MATCH"
    CATEGORY_MATCH = "CATEGORY_MATCH"
    FLEXIBLE_INSTALLMENT_MATCH = "FLEXIBLE_INSTALLMENT_MATCH"
    STABLE_FIXED_INSTALLMENT_MATCH = "STABLE_FIXED_INSTALLMENT_MATCH"
    JOB_TERM_MATCH = "JOB_TERM_MATCH"


@dataclass(frozen=True)
class ScoreEvidence:
    evidence_type: EvidenceType
    source_field: str
    normalized_key: str
    points: float


@dataclass
class CandidateScore:
    product: dict[str, Any]
    selected_option: dict[str, Any]
    base_rate: Decimal
    term_months: int
    financial_benefit_score: float = 0.0
    semantic_relevance_score: float = 0.0
    financial_suitability_score: float = 0.0
    condition_match_score: float = 0.0
    personalized_score: float = 0.0
    evidences: list[ScoreEvidence] = field(default_factory=list)


CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    ExpenseCategory.FOOD.value: ("음식", "외식", "식비"),
    ExpenseCategory.TRANSPORTATION.value: ("교통", "대중교통"),
    ExpenseCategory.HOUSING.value: ("주거", "주택", "월세"),
    ExpenseCategory.COMMUNICATION.value: ("통신", "휴대폰"),
    ExpenseCategory.MEDICAL.value: ("의료", "병원", "건강"),
    ExpenseCategory.EDUCATION.value: ("교육", "학원", "장학"),
    ExpenseCategory.SHOPPING.value: ("쇼핑", "백화점"),
    ExpenseCategory.LEISURE.value: ("여행", "항공", "숙박", "레저"),
    ExpenseCategory.INSURANCE.value: ("보험",),
    ExpenseCategory.FINANCE.value: ("금융", "대출", "연금"),
}

# The project has no job-name enum. Only these controlled concepts may create
# structured evidence; every other jobName is semantic-search context only.
CONTROLLED_JOB_CONCEPTS: dict[str, tuple[str, ...]] = {
    "SALARY": ("직장인", "회사원", "근로자"),
    "BUSINESS": ("개인사업자", "사업자", "자영업"),
    "FREELANCE": ("프리랜서",),
}

SOURCE_TEXT_FIELDS = (
    "summary",
    "target_group",
    "join_way",
    "special_conditions_text",
    "eligible_members_text",
    "notes_text",
)


def percentile_ranks(values: list[float]) -> list[float]:
    """Weak percentile ranks: equal values always receive equal scores."""
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    ordered = sorted(values)
    return [sum(item < value for item in ordered) / (len(values) - 1) for value in values]


class PersonalizedProductScoringService:
    def __init__(self, policy: PersonalizedScorePolicy | None = None):
        self.policy = policy or PersonalizedScorePolicy()

    def score_all(
        self,
        products: list[dict[str, Any]],
        profile: RecommendationUserProfile,
        semantic_similarities: dict[str, float] | None = None,
    ) -> list[CandidateScore]:
        candidates = [self._candidate(item) for item in products if item.get("product_type") in {"SAVINGS", "DEPOSIT"}]
        candidates = [item for item in candidates if item is not None]
        if profile.investment_budget <= 0 or not candidates:
            return []
        benefit = percentile_ranks([float(item.base_rate) for item in candidates])
        similarities = semantic_similarities or {}
        raw_semantic = [float(similarities.get(item.product["product_id"], 0.0)) for item in candidates]
        semantic = percentile_ranks(raw_semantic) if profile.has_semantic_signals else [0.0] * len(candidates)
        for item, benefit_score, semantic_score in zip(candidates, benefit, semantic):
            item.financial_benefit_score = benefit_score
            item.semantic_relevance_score = semantic_score
            item.financial_suitability_score = self._financial_suitability(item, profile)
            item.evidences = self._condition_evidence(item, profile)
            item.condition_match_score = min(1.0, sum(ev.points for ev in item.evidences))
            item.personalized_score = float(
                Decimal(str(item.financial_benefit_score)) * self.policy.financial_benefit_weight
                + Decimal(str(item.semantic_relevance_score)) * self.policy.semantic_relevance_weight
                + Decimal(str(item.financial_suitability_score)) * self.policy.financial_suitability_weight
                + Decimal(str(item.condition_match_score)) * self.policy.condition_match_weight
            )
            for value in (
                item.financial_benefit_score,
                item.semantic_relevance_score,
                item.financial_suitability_score,
                item.condition_match_score,
                item.personalized_score,
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError("Personalized scores must be within [0, 1]")
        return sorted(
            candidates,
            key=lambda item: (
                -item.personalized_score,
                -float(item.base_rate),
                item.term_months,
                0 if item.product["product_type"] == "SAVINGS" else 1,
                item.product["product_id"],
            ),
        )

    def _candidate(self, product: dict[str, Any]) -> CandidateScore | None:
        details = product.get("details") or {}
        options = details.get("options") or []
        valid = []
        for option in options:
            try:
                rate = Decimal(str(option.get("base_interest_rate")))
                preferred_raw = option.get("preferred_interest_rate")
                preferred = None if preferred_raw is None else Decimal(str(preferred_raw))
                term = int(option.get("term_months"))
                if not rate.is_finite() or rate < 0 or term <= 0:
                    continue
                if preferred is not None and (not preferred.is_finite() or preferred < rate):
                    continue
                valid.append((rate, term, self._option_key(option), option))
            except (TypeError, ValueError, ArithmeticError):
                continue
        if not valid:
            return None
        rate, term, _, option = sorted(valid, key=lambda value: (-value[0], value[1], value[2]))[0]
        enriched = dict(product)
        enriched_details = dict(details)
        enriched_details.update({
            "selected_option": option,
            "interest_rate": float(rate),
            "preferred_interest_rate": option.get("preferred_interest_rate"),
            "term_months": term,
        })
        enriched["details"] = enriched_details
        return CandidateScore(enriched, option, rate, term)

    def select_option(self, product: dict[str, Any]) -> CandidateScore | None:
        """Public deterministic option-validation boundary for serving promotion."""
        return self._candidate(product)

    @staticmethod
    def _option_key(option):
        return json.dumps(option, ensure_ascii=False, sort_keys=True, default=str)

    def _financial_suitability(self, item: CandidateScore, profile: RecommendationUserProfile) -> float:
        cash_capacity = min(1.0, max(0.0, profile.available_ratio / 0.3))
        if item.product["product_type"] == "DEPOSIT":
            return cash_capacity  # one-time MVP allocation; no recurring obligation
        reserve = str(item.selected_option.get("reserve_type") or "").upper()
        if reserve == "S":
            return max(0.0, min(1.0, profile.stable_contribution_capacity))
        if reserve == "F":
            return max(0.0, min(1.0, (cash_capacity + profile.flexible_contribution_need) / 2.0))
        return 0.0

    def _condition_evidence(self, item: CandidateScore, profile: RecommendationUserProfile) -> list[ScoreEvidence]:
        texts = self._source_texts(item.product)
        points = float(self.policy.condition_evidence_points)
        found: dict[tuple[EvidenceType, str], ScoreEvidence] = {}

        def add(kind: EvidenceType, field_name: str, key: str):
            found.setdefault((kind, key), ScoreEvidence(kind, field_name, key, points))

        if profile.multiple_income_sources:
            hit = self._find_term(texts, ("n잡", "복수소득", "복수 소득"))
            if hit:
                add(EvidenceType.MULTI_INCOME_MATCH, hit[0], "MULTI_INCOME")
        controlled = self._controlled_job_concepts(profile.job_names)
        if "SALARY" in controlled:
            hit = self._find_term(texts, ("급여", "직장인", "근로자"))
            if hit:
                add(EvidenceType.SALARY_CONDITION_MATCH, hit[0], "SALARY")
        if "BUSINESS" in controlled:
            hit = self._find_term(texts, ("사업소득", "개인사업자", "사업자"))
            if hit:
                add(EvidenceType.BUSINESS_INCOME_MATCH, hit[0], "BUSINESS")
        if "FREELANCE" in controlled:
            hit = self._find_term(texts, ("프리랜서",))
            if hit:
                add(EvidenceType.JOB_TERM_MATCH, hit[0], "FREELANCE")
        for category in profile.categories:
            terms = CATEGORY_TERMS.get(category.category.upper())
            hit = self._find_term(texts, terms or ())
            if hit:
                add(EvidenceType.CATEGORY_MATCH, hit[0], category.category.upper())
        reserve = str(item.selected_option.get("reserve_type") or "").upper()
        if reserve == "F" and profile.flexible_contribution_need > 0:
            add(EvidenceType.FLEXIBLE_INSTALLMENT_MATCH, "options.reserve_type", "F")
        if reserve == "S" and profile.stable_contribution_capacity >= 0.5:
            add(EvidenceType.STABLE_FIXED_INSTALLMENT_MATCH, "options.reserve_type", "S")
        return sorted(found.values(), key=lambda value: (value.evidence_type, value.normalized_key))

    @staticmethod
    def _controlled_job_concepts(job_names: tuple[str, ...]) -> set[str]:
        result = set()
        for name in job_names:
            normalized = " ".join(name.lower().split())
            for concept, aliases in CONTROLLED_JOB_CONCEPTS.items():
                if normalized in aliases:
                    result.add(concept)
        return result

    @staticmethod
    def _source_texts(product: dict[str, Any]) -> dict[str, str]:
        details = product.get("details") or {}
        result = {}
        for field_name in SOURCE_TEXT_FIELDS:
            value = product.get(field_name) if field_name in product else details.get(field_name)
            if isinstance(value, str) and value.strip():
                result[field_name] = " ".join(value.lower().split())
        return result

    @staticmethod
    def _find_term(texts: dict[str, str], terms: tuple[str, ...]):
        for field_name, text in texts.items():
            for term in terms:
                if term.lower() in text:
                    return field_name, term
        return None
