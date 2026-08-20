import json

import pytest

from app.commands.transform_finlife_products import main
from app.schemas.finlife_product import ProductType
from app.services.finlife_collection_service import (
    FinlifeCollectionError,
    FinlifeCollectionErrorKind,
    FinlifeCollectionOutcome,
    FinlifeCollectionSummary,
    FinlifeProductTypeSummary,
)


def summary():
    by_type = [
        FinlifeProductTypeSummary(
            product_type=product_type,
            page_count=1,
            base_count=1,
            option_count=1,
            source_count=1,
            mapped_count=1,
            valid_count=1,
            invalid_count=0,
            review_required_count=0,
            orphan_option_count=0,
            optionless_product_count=0,
            duplicate_removed_count=0,
            snapshot_complete=True,
        )
        for product_type in (ProductType.SAVINGS, ProductType.DEPOSIT)
    ]
    return FinlifeCollectionSummary(
        success=True,
        source_count=2,
        mapped_count=2,
        valid_count=2,
        invalid_count=0,
        review_required_count=0,
        orphan_option_count=0,
        optionless_product_count=0,
        duplicate_removed_count=0,
        canonical_hash_success_count=2,
        canonical_hash_failure_count=0,
        knowledge_id_duplicate_count=0,
        snapshot_complete=True,
        by_product_type=by_type,
    )


class Service:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def collect_and_transform(self, group_no):
        self.calls.append(group_no)
        if self.error:
            raise self.error
        return FinlifeCollectionOutcome(products=[], summary=summary())


def test_a_cli_outputs_only_safe_aggregate(capsys):
    service = Service()
    code = main(["--top-fin-group-no", "020000", "--dry-run"], service=service)
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert service.calls == ["020000"]
    assert output["mapped_count"] == 2
    assert output["canonical_hash_success_count"] == 2
    serialized = json.dumps(output)
    for forbidden in ("product_name", "provider", "details", "document", "content_hash"):
        assert forbidden not in serialized


def test_a_cli_rejects_page_limit_and_requires_dry_run():
    service = Service()
    with pytest.raises(SystemExit):
        main(["--dry-run", "--page-limit", "1"], service=service)
    with pytest.raises(SystemExit):
        main([], service=service)
    assert service.calls == []


def test_a_cli_reports_only_safe_error_kind(capsys):
    service = Service(FinlifeCollectionError(
        FinlifeCollectionErrorKind.SOURCE_COLLECTION_FAILED,
        source_error_kind="RATE_LIMITED",
    ))
    code = main(["--dry-run"], service=service)
    output = json.loads(capsys.readouterr().out)
    assert code == 1
    assert output == {
        "success": False,
        "snapshot_complete": False,
        "error_counts": {"RATE_LIMITED": 1},
    }
