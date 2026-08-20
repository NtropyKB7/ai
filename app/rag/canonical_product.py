from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from decimal import Decimal

from app.schemas.product_knowledge import RawFinancialProduct


_HASH_EXCLUDED_DETAIL_FIELDS = {"collected_at", "source_submitted_at"}


def _canonicalize(value):
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, dict):
        return {
            key: _canonicalize(item)
            for key, item in sorted(value.items())
            if key not in _HASH_EXCLUDED_DETAIL_FIELDS
        }
    if isinstance(value, list):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return value


def canonical_product_bytes(product: RawFinancialProduct) -> bytes:
    payload = deepcopy(product.model_dump(mode="json"))
    payload.pop("summary", None)
    payload.pop("target_group", None)
    payload.pop("njob_trend_tip", None)
    canonical = _canonicalize(payload)
    return json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def product_content_hash(product: RawFinancialProduct) -> str:
    return hashlib.sha256(canonical_product_bytes(product)).hexdigest()
