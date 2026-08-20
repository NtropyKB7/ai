from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from app.schemas.product import CategoryExpenseSummary, ProductRecommendationRequest


INVESTMENT_BUDGET_RATIO = Decimal("0.5")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RecommendationUserProfile:
    investment_budget: int
    available_ratio: float
    expense_ratio: float
    income_trend: str
    income_change_rate: float | None
    income_volatility: float
    job_count: int
    income_job_count: int
    income_concentration: float | None
    multiple_income_sources: bool
    total_work_minutes: int
    total_work_days: int
    average_fatigue: float | None
    latest_fatigue: float | None
    job_names: tuple[str, ...]
    categories: tuple[CategoryExpenseSummary, ...]
    stable_contribution_capacity: float
    flexible_contribution_need: float
    has_semantic_signals: bool

    def search_document(self) -> str:
        parts = [
            f"소득상태 {self.income_trend}",
            f"소득변동성 {self.income_volatility:.4f}",
            f"투자예산 {self.investment_budget}원",
            f"소득원수 {self.income_job_count}",
            f"복수소득원 {'예' if self.multiple_income_sources else '아니오'}",
        ]
        if self.job_names:
            parts.append("잡 " + ", ".join(self.job_names))
        if self.total_work_minutes or self.total_work_days:
            parts.append(
                f"총근무분 {self.total_work_minutes}, 총근무일 {self.total_work_days}"
            )
        if self.average_fatigue is not None or self.latest_fatigue is not None:
            parts.append(
                f"평균피로 {self.average_fatigue}, 최근피로 {self.latest_fatigue}"
            )
        if self.categories:
            parts.append(
                "소비 "
                + ", ".join(
                    f"{item.category}:{item.amount or 0}원:{item.ratio or 0:.4f}"
                    for item in self.categories
                )
            )
        parts.append(
            "안정적 정기납입 적합"
            if self.stable_contribution_capacity >= 0.5
            else "유연한 납입 필요"
        )
        return " | ".join(parts)


class RecommendationProfileService:
    AVAILABLE_CAPACITY_RATIO = 0.3
    HIGH_VOLATILITY_THRESHOLD = 0.2

    def build(
        self,
        request: ProductRecommendationRequest,
        categories: list[CategoryExpenseSummary],
    ) -> RecommendationUserProfile:
        income = max(request.total_income or 0, 0)
        expense = max(request.total_expense or 0, 0)
        funds = max(request.available_funds or 0, 0)
        budget = int(
            (Decimal(funds) * INVESTMENT_BUDGET_RATIO).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        available_ratio = funds / income if income else 0.0
        expense_ratio = expense / income if income else 0.0
        change_rate = self._income_change_rate(request)
        trend = (
            "UNKNOWN" if change_rate is None else
            "INCREASING" if change_rate > 0 else
            "DECREASING" if change_rate < 0 else "STABLE"
        )
        volatility = max(float(request.income_volatility or 0.0), 0.0)
        jobs = list(request.job_insight_inputs or [])
        positive_jobs = [item for item in jobs if (item.incomeAmount or 0) > 0]
        ratios = [float(item.incomeRatio) for item in positive_jobs if item.incomeRatio is not None]
        amounts = [max(item.incomeAmount or 0, 0) for item in positive_jobs]
        concentration = None
        if ratios:
            concentration = max(ratios)
        elif sum(amounts) > 0:
            concentration = max(amounts) / sum(amounts)
        average_values = [float(item.averageFatigue) for item in jobs if item.averageFatigue is not None]
        latest_values = [float(item.latestFatigue) for item in jobs if item.latestFatigue is not None]
        cash_capacity = _clamp(available_ratio / self.AVAILABLE_CAPACITY_RATIO)
        stability = 1.0 - _clamp(volatility / self.HIGH_VOLATILITY_THRESHOLD)
        trend_health = 0.5 if change_rate is None or change_rate == 0 else (1.0 if change_rate > 0 else 0.0)
        stable_capacity = (cash_capacity + stability + trend_health) / 3.0
        flexible_need = max(1.0 - stability, 1.0 if change_rate is not None and change_rate < 0 else 0.0)
        normalized_categories = tuple(sorted(categories, key=lambda item: item.category))
        job_names = tuple(sorted({item.jobName.strip() for item in jobs if item.jobName and item.jobName.strip()}))
        has_signals = bool(
            income > 0
            or funds > 0
            or jobs
            or normalized_categories
            or request.income_volatility is not None
            or change_rate is not None
        )
        return RecommendationUserProfile(
            investment_budget=budget,
            available_ratio=available_ratio,
            expense_ratio=expense_ratio,
            income_trend=trend,
            income_change_rate=change_rate,
            income_volatility=volatility,
            job_count=len(jobs),
            income_job_count=len(positive_jobs),
            income_concentration=concentration,
            multiple_income_sources=len(positive_jobs) >= 2,
            total_work_minutes=sum(max(item.totalWorkMinutes or 0, 0) for item in jobs),
            total_work_days=sum(max(item.workDays or 0, 0) for item in jobs),
            average_fatigue=(sum(average_values) / len(average_values) if average_values else None),
            latest_fatigue=(max(latest_values) if latest_values else None),
            job_names=job_names,
            categories=normalized_categories,
            stable_contribution_capacity=_clamp(stable_capacity),
            flexible_contribution_need=_clamp(flexible_need),
            has_semantic_signals=has_signals,
        )

    @staticmethod
    def _income_change_rate(request: ProductRecommendationRequest) -> float | None:
        if request.income_change_rate is not None:
            return float(request.income_change_rate)
        previous = request.previous_month_income
        if previous and request.income_change_amount is not None:
            return float(request.income_change_amount) / previous
        if previous:
            return float((request.total_income or 0) - previous) / previous
        return None
