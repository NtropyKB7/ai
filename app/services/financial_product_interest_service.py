from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR


class FinancialProductInterestService:
    """Gross-interest MVP calculations; only the final won amount is floored."""

    @staticmethod
    def monthly_interest(
        product_type: str,
        investment_budget: int,
        annual_base_rate: Decimal,
        term_months: int,
    ) -> int:
        if investment_budget <= 0 or term_months <= 0:
            raise ValueError("A positive budget and term are required")
        rate = Decimal(str(annual_base_rate))
        if not rate.is_finite() or rate < 0:
            raise ValueError("A finite non-negative base rate is required")
        budget = Decimal(investment_budget)
        months = Decimal(term_months)
        annual_fraction = rate / Decimal("100")
        if product_type == "SAVINGS":
            # Beginning-of-month approximation: each monthly contribution earns
            # interest for N, N-1, ... 1 months.
            total = (
                budget
                * annual_fraction
                / Decimal("12")
                * months
                * (months + Decimal("1"))
                / Decimal("2")
            )
        elif product_type == "DEPOSIT":
            # MVP principal is this month's allocated surplus, not total wealth.
            total = budget * annual_fraction * months / Decimal("12")
        else:
            raise ValueError("Interest is supported for SAVINGS or DEPOSIT only")
        return int((total / months).to_integral_value(rounding=ROUND_FLOOR))
