from datetime import datetime, timezone

import pytest

from app.rag.finlife_product_mapper import FinlifeMappingError, FinlifeProductMapper
from app.rag.finlife_response_parser import FinlifeResponseError, parse_finlife_response
from app.schemas.finlife_product import ProductType
from finlife_fixtures import response


@pytest.mark.parametrize("product_type", [ProductType.SAVINGS, ProductType.DEPOSIT])
def test_parses_official_response_shapes(product_type):
    page = parse_finlife_response(response(product_type.value), product_type)
    assert page.total_count == 1
    assert page.max_page_no == 1
    assert page.now_page_no == 1


def test_normalizes_numeric_pagination_values():
    payload = response()
    payload["result"].update(total_count=1, max_page_no=1, now_page_no=1)
    assert parse_finlife_response(payload, ProductType.SAVINGS).total_count == 1


def test_distinguishes_error_from_empty_success():
    payload = response(option=False, total="0")
    payload["result"].update(err_cd="020", err_msg="일일검색 허용횟수 초과", baseList=[])
    with pytest.raises(FinlifeResponseError) as error:
        parse_finlife_response(payload, ProductType.SAVINGS)
    assert error.value.code == "020"


def test_accepts_normal_empty_success():
    payload = response(option=False, total="0")
    payload["result"].update(baseList=[], optionList=[])
    assert parse_finlife_response(payload, ProductType.SAVINGS).bases == []


def test_rejects_product_division_mismatch():
    with pytest.raises(ValueError, match="prdt_div"):
        parse_finlife_response(response("DEPOSIT"), ProductType.SAVINGS)


def test_deposit_rejects_savings_only_reserve_fields():
    payload = response("DEPOSIT")
    payload["result"]["optionList"][0].update(rsrv_type="F", rsrv_type_nm="자유적립식")
    with pytest.raises(Exception, match="rsrv_type"):
        parse_finlife_response(payload, ProductType.DEPOSIT)


def _map(payload, product_type):
    page = parse_finlife_response(payload, product_type)
    return FinlifeProductMapper().combine_pages(
        [page], product_type, datetime(2026, 8, 20, tzinfo=timezone.utc)
    )


def test_maps_savings_with_reserve_type_and_nullable_fields():
    product = _map(response("SAVINGS"), ProductType.SAVINGS)[0]
    assert product.product_id == "FSS_FINLIFE:SAVINGS:0010001:P001"
    assert product.summary is None
    assert product.details["options"][0]["reserve_type"] == "F"
    assert product.details["options"][0]["base_interest_rate"] is None
    assert product.details["source_region_code"] == "020000"
    assert product.details["max_limit"] is None
    assert product.details["disclosure_end_date"] is None


def test_maps_deposit_without_reserve_type_or_invented_fields():
    product = _map(response("DEPOSIT"), ProductType.DEPOSIT)[0]
    details = product.details
    assert "reserve_type" not in details["options"][0]
    assert "sale_start_date" not in details
    assert "sale_end_date" not in details
    assert "minimum_deposit" not in details
    assert details["disclosure_start_date"] == "20260801"
    assert details["source_submitted_at"] == "202608011200"
    assert details["collected_at"] == "2026-08-20T00:00:00Z"


def test_deduplicates_identical_base_and_option_rows():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    page.bases.append(page.bases[0].model_copy(deep=True))
    page.options.append(page.options[0].model_copy(deep=True))
    product = FinlifeProductMapper().combine_pages([page], ProductType.SAVINGS)[0]
    assert len(product.details["options"]) == 1


def test_rejects_conflicting_base_rows():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    page.bases.append(page.bases[0].model_copy(update={"fin_prdt_nm": "충돌"}))
    with pytest.raises(FinlifeMappingError, match="Conflicting base"):
        FinlifeProductMapper().combine_pages([page], ProductType.SAVINGS)


def test_rejects_orphan_option_rows():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    page.options[0].fin_prdt_cd = "ORPHAN"
    with pytest.raises(FinlifeMappingError, match="Orphan"):
        FinlifeProductMapper().combine_pages([page], ProductType.SAVINGS)


def test_rejects_conflicting_option_rows():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    page.options.append(page.options[0].model_copy(update={"intr_rate2": "9.9"}))
    with pytest.raises(FinlifeMappingError, match="Conflicting option"):
        FinlifeProductMapper().combine_pages([page], ProductType.SAVINGS)


def test_preserves_product_without_options():
    product = _map(response(option=False), ProductType.SAVINGS)[0]
    assert product.details["options"] == []


def test_isolates_one_product_option_conflict_and_keeps_valid_products():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    second_base = page.bases[0].model_copy(update={"fin_prdt_cd": "P002"})
    second_option = page.options[0].model_copy(update={"fin_prdt_cd": "P002"})
    page.bases.append(second_base)
    page.options.extend([
        second_option,
        second_option.model_copy(update={"intr_rate2": "9.9"}),
    ])
    result = FinlifeProductMapper().combine_pages_with_issues(
        [page], ProductType.SAVINGS
    )
    assert [item.product_id for item in result.products] == [
        "FSS_FINLIFE:SAVINGS:0010001:P001"
    ]
    assert result.failed_count == 1


def test_isolates_orphan_option_as_invalid_count():
    page = parse_finlife_response(response(), ProductType.SAVINGS)
    page.options.append(page.options[0].model_copy(update={"fin_prdt_cd": "ORPHAN"}))
    result = FinlifeProductMapper().combine_pages_with_issues(
        [page], ProductType.SAVINGS
    )
    assert len(result.products) == 1
    assert result.failed_count == 1


def test_options_missing_is_review_required_not_failed():
    page = parse_finlife_response(response(option=False), ProductType.SAVINGS)
    result = FinlifeProductMapper().combine_pages_with_issues(
        [page], ProductType.SAVINGS
    )
    assert len(result.products) == 1
    assert result.failed_count == 0
    assert result.review_required_count == 1
