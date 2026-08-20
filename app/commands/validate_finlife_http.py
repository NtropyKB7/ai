from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.core.config import settings
from app.rag.finlife_http_source import (
    FinlifeHttpFinancialProductSource,
    FinlifeHttpSourceError,
    FinlifeHttpTimeouts,
)
from app.schemas.finlife_product import ProductType


def _source() -> FinlifeHttpFinancialProductSource:
    return FinlifeHttpFinancialProductSource(
        api_key=settings.FINLIFE_API_KEY,
        base_url=settings.FINLIFE_BASE_URL,
        endpoint_paths={
            ProductType.SAVINGS: settings.FINLIFE_SAVINGS_PATH,
            ProductType.DEPOSIT: settings.FINLIFE_DEPOSIT_PATH,
        },
        timeouts=FinlifeHttpTimeouts(
            connect=settings.FINLIFE_CONNECT_TIMEOUT_SECONDS,
            read=settings.FINLIFE_READ_TIMEOUT_SECONDS,
            write=settings.FINLIFE_WRITE_TIMEOUT_SECONDS,
            pool=settings.FINLIFE_POOL_TIMEOUT_SECONDS,
        ),
    )


def main(argv=None, *, source=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Finlife HTTP response structure without staging writes"
    )
    parser.add_argument(
        "--product-type",
        required=True,
        choices=(ProductType.SAVINGS.value, ProductType.DEPOSIT.value),
    )
    parser.add_argument("--top-fin-group-no", default="020000")
    parser.add_argument("--page-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)

    try:
        result = (source or _source()).fetch(
            ProductType(args.product_type),
            top_fin_group_no=args.top_fin_group_no,
            page_limit=args.page_limit,
        )
    except FinlifeHttpSourceError as exc:
        print(json.dumps({
            "success": False,
            "http_status_classification": exc.kind.value,
            "status_code": exc.status_code,
            "finlife_error_code": exc.finlife_error_code,
        }, ensure_ascii=False, sort_keys=True))
        return 1

    output = {
        "success": True,
        "product_type": result.product_type.value,
        "endpoint": {
            "host": result.endpoint_host,
            "path": result.endpoint_path,
        },
        "http_status_classification": "SUCCESS",
        "incomplete_snapshot": not result.completeness.is_complete,
        "page_limited": result.page_limited,
        "pages": [asdict(item) for item in result.inspections],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
