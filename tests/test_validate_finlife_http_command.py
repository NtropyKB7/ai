import json

from app.commands.validate_finlife_http import main
from app.rag.finlife_http_source import (
    FinlifeFetchResult,
    FinlifeHttpErrorKind,
    FinlifeHttpSourceError,
    FinlifePageInspection,
)
from app.schemas.finlife_product import ProductType, SnapshotCompleteness


class StubSource:
    def __init__(self, error=None):
        self.error = error
        self.page_limits = []

    def fetch(self, product_type, *, top_fin_group_no, page_limit):
        self.page_limits.append(page_limit)
        if self.error:
            raise self.error
        return FinlifeFetchResult(
            product_type=product_type,
            endpoint_host="finlife.fss.or.kr",
            endpoint_path="/finlifeapi/savingProductsSearch.json",
            pages=[],
            completeness=(
                SnapshotCompleteness()
                if page_limit is None
                else SnapshotCompleteness(all_pages_succeeded=False)
            ),
            inspections=[FinlifePageInspection(
                page_no=1,
                http_status=200,
                result_wrapper_present=True,
                base_list_present=True,
                option_list_present=True,
                prdt_div_present=False,
                err_cd="000",
                total_count=10,
                max_page_no=2,
                now_page_no=1,
                base_count=5,
                option_count=8,
                base_fields_present=("fin_prdt_cd",),
                base_fields_nullable=("dcls_end_day",),
                option_fields_present=("save_trm",),
                option_fields_nullable=(),
            )],
            page_limited=page_limit is not None,
        )


def test_cli_reports_safe_structural_dry_run(capsys):
    code = main([
        "--product-type", "SAVINGS",
        "--top-fin-group-no", "020000",
        "--page-limit", "1",
        "--dry-run",
    ], source=StubSource())
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["incomplete_snapshot"] is True
    assert output["endpoint"] == {
        "host": "finlife.fss.or.kr",
        "path": "/finlifeapi/savingProductsSearch.json",
    }
    assert output["pages"][0]["base_count"] == 5
    assert "auth" not in json.dumps(output).lower()


def test_cli_reports_safe_classified_error(capsys):
    error = FinlifeHttpSourceError(
        FinlifeHttpErrorKind.RATE_LIMITED, status_code=429
    )
    code = main([
        "--product-type", "DEPOSIT",
        "--page-limit", "1",
        "--dry-run",
    ], source=StubSource(error))
    output = json.loads(capsys.readouterr().out)
    assert code == 1
    assert output == {
        "success": False,
        "http_status_classification": "RATE_LIMITED",
        "status_code": 429,
        "finlife_error_code": None,
    }


def test_cli_without_page_limit_requests_complete_snapshot(capsys):
    source = StubSource()
    code = main([
        "--product-type", "SAVINGS",
        "--top-fin-group-no", "020000",
        "--dry-run",
    ], source=source)
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert source.page_limits == [None]
    assert output["incomplete_snapshot"] is False
    assert output["page_limited"] is False
