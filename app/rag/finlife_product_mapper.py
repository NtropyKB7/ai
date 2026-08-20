from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

from app.schemas.finlife_product import (
    DepositOptionDetails,
    DepositProductDetails,
    FinlifeBaseInfo,
    FinlifeDepositOption,
    FinlifePage,
    FinlifeSavingsOption,
    ProductType,
    SavingsOptionDetails,
    SavingsProductDetails,
)
from app.schemas.product_knowledge import RawFinancialProduct
from app.schemas.product_sync import (
    MappingIssue,
    ProductMappingResult,
    ValidationStatus,
)


class FinlifeMappingError(ValueError):
    pass


def _text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized or None


def _term(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def build_knowledge_id(product_type: ProductType, base: FinlifeBaseInfo) -> str:
    if product_type not in (ProductType.SAVINGS, ProductType.DEPOSIT):
        raise FinlifeMappingError("Finlife knowledge_id requires SAVINGS or DEPOSIT")
    return f"FSS_FINLIFE:{product_type.value}:{base.fin_co_no}:{base.fin_prdt_cd}"


class FinlifeProductMapper:
    def combine_pages(
        self,
        pages: list[FinlifePage],
        product_type: ProductType,
        collected_at: datetime | None = None,
    ) -> list[RawFinancialProduct]:
        result = self.combine_pages_with_issues(pages, product_type, collected_at)
        invalid_issue = next(
            (
                issue
                for issue in result.issues
                if issue.validation_status == ValidationStatus.INVALID
            ),
            None,
        )
        if invalid_issue:
            raise FinlifeMappingError(invalid_issue.message)
        return result.products

    def combine_pages_with_issues(
        self,
        pages: list[FinlifePage],
        product_type: ProductType,
        collected_at: datetime | None = None,
    ) -> ProductMappingResult:
        if product_type not in (ProductType.SAVINGS, ProductType.DEPOSIT):
            raise FinlifeMappingError("Only SAVINGS and DEPOSIT can be mapped")
        if any(page.product_type != product_type for page in pages):
            raise FinlifeMappingError("prdt_div and requested product type do not match")

        bases: dict[tuple[str, str, str], FinlifeBaseInfo] = {}
        options = defaultdict(list)
        for page in pages:
            for base in page.bases:
                previous = bases.get(base.join_key)
                if previous and previous.model_dump() != base.model_dump():
                    raise FinlifeMappingError(f"Conflicting base row: {base.join_key}")
                bases[base.join_key] = base
            for option in page.options:
                options[option.join_key].append(option)

        captured_at = collected_at or datetime.now(timezone.utc)
        issues = [
            MappingIssue(
                knowledge_id=self._knowledge_id_from_key(product_type, key),
                validation_status=ValidationStatus.INVALID,
                message=f"Orphan option rows: {key}",
            )
            for key in sorted(set(options) - set(bases))
        ]
        products = []
        for _, base in sorted(bases.items()):
            knowledge_id = build_knowledge_id(product_type, base)
            if not all((_text(base.fin_prdt_nm), _text(base.kor_co_nm))):
                issues.append(MappingIssue(
                    knowledge_id=knowledge_id,
                    validation_status=ValidationStatus.INVALID,
                    message=f"Missing required product field: {knowledge_id}",
                ))
                continue
            try:
                product = self._to_raw(
                    product_type, base, options[base.join_key], captured_at
                )
                product.model_dump_json()
            except Exception as exc:
                issues.append(MappingIssue(
                    knowledge_id=knowledge_id,
                    validation_status=ValidationStatus.INVALID,
                    message=f"Product mapping failed: {knowledge_id}: {exc}",
                ))
                continue
            products.append(product)
            if not product.details.get("options"):
                issues.append(MappingIssue(
                    knowledge_id=knowledge_id,
                    validation_status=ValidationStatus.REVIEW_REQUIRED,
                    message=f"Product has no options: {knowledge_id}",
                ))
        return ProductMappingResult(
            source_count=len(bases) + len(set(options) - set(bases)),
            products=products,
            issues=issues,
        )

    @staticmethod
    def _knowledge_id_from_key(product_type, key):
        _, fin_co_no, fin_prdt_cd = key
        return f"FSS_FINLIFE:{product_type.value}:{fin_co_no}:{fin_prdt_cd}"

    def _to_raw(self, product_type, base, options, collected_at):
        typed_options = self._deduplicate_options(product_type, options)
        common = dict(
            source_product_id=base.fin_prdt_cd,
            source_provider_id=base.fin_co_no,
            disclosure_month=base.dcls_month,
            disclosure_start_date=base.dcls_strt_day,
            disclosure_end_date=base.dcls_end_day,
            source_submitted_at=base.fin_co_subm_day,
            collected_at=collected_at,
            join_way=_text(base.join_way),
            maturity_interest_text=_text(base.mtrt_int),
            special_conditions_text=_text(base.spcl_cnd),
            join_restriction_code=_text(base.join_deny),
            eligible_members_text=_text(base.join_member),
            notes_text=_text(base.etc_note),
            max_limit=base.max_limit,
            options=typed_options,
        )
        details = (
            SavingsProductDetails(**common)
            if product_type == ProductType.SAVINGS
            else DepositProductDetails(**common)
        )
        return RawFinancialProduct(
            product_id=build_knowledge_id(product_type, base),
            product_name=_text(base.fin_prdt_nm) or "",
            product_type=product_type.value,
            provider=_text(base.kor_co_nm) or "",
            summary=None,
            details=details.model_dump(mode="json"),
        )

    def _deduplicate_options(self, product_type, rows):
        logical: dict[tuple, object] = {}
        for row in rows:
            if product_type == ProductType.SAVINGS:
                if not isinstance(row, FinlifeSavingsOption):
                    raise FinlifeMappingError("Savings option is missing reserve type schema")
                key = (row.intr_rate_type, row.rsrv_type, row.save_trm)
                mapped = SavingsOptionDetails(
                    interest_rate_type=row.intr_rate_type,
                    interest_rate_type_name=row.intr_rate_type_nm,
                    reserve_type=row.rsrv_type,
                    reserve_type_name=row.rsrv_type_nm,
                    term_months=_term(row.save_trm),
                    base_interest_rate=row.intr_rate,
                    preferred_interest_rate=row.intr_rate2,
                )
            else:
                if not isinstance(row, FinlifeDepositOption):
                    raise FinlifeMappingError("Deposit option contains savings-only fields")
                key = (row.intr_rate_type, row.save_trm)
                mapped = DepositOptionDetails(
                    interest_rate_type=row.intr_rate_type,
                    interest_rate_type_name=row.intr_rate_type_nm,
                    term_months=_term(row.save_trm),
                    base_interest_rate=row.intr_rate,
                    preferred_interest_rate=row.intr_rate2,
                )
            previous = logical.get(key)
            if previous is not None and previous != mapped:
                raise FinlifeMappingError(f"Conflicting option row: {key}")
            logical[key] = mapped
        return [logical[key] for key in sorted(logical, key=lambda item: tuple(x or "" for x in item))]
