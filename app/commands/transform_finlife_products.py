from __future__ import annotations

import argparse
import json

from app.commands.validate_finlife_http import _source
from app.services.finlife_collection_service import (
    FinlifeCollectionError,
    FinlifeCollectionService,
)


def main(argv=None, *, service=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect and transform complete Finlife snapshots without storage"
    )
    parser.add_argument("--top-fin-group-no", default="020000")
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)

    runner = service or FinlifeCollectionService(_source())
    try:
        outcome = runner.collect_and_transform(args.top_fin_group_no)
    except FinlifeCollectionError as exc:
        error_kind = exc.source_error_kind or exc.kind.value
        print(json.dumps({
            "success": False,
            "snapshot_complete": False,
            "error_counts": {error_kind: 1},
        }, ensure_ascii=False, sort_keys=True))
        return 1

    print(json.dumps(
        outcome.summary.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0 if outcome.summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
