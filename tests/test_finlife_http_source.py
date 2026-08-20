import json
import logging
import ssl

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.rag.finlife_http_source import (
    FinlifeHttpErrorKind,
    FinlifeHttpFinancialProductSource,
    FinlifeHttpSourceError,
    FinlifeHttpTimeouts,
)
from app.schemas.finlife_product import ProductType


SYNTHETIC_KEY = "synthetic-finlife-unit-key"


def payload(product_type=ProductType.SAVINGS, page=1, total=1, maximum=1):
    savings = product_type == ProductType.SAVINGS
    option = {
        "dcls_month": "202608",
        "fin_co_no": "0010001",
        "fin_prdt_cd": f"P{page}",
        "intr_rate_type": "S",
        "intr_rate_type_nm": "단리",
        "save_trm": "12",
        "intr_rate": 2.0,
        "intr_rate2": None,
    }
    if savings:
        option.update(rsrv_type="S", rsrv_type_nm="정액적립식")
    return {
        "result": {
            "prdt_div": "S" if savings else "D",
            "err_cd": "000",
            "err_msg": "정상",
            "total_count": str(total),
            "max_page_no": maximum,
            "now_page_no": str(page),
            "baseList": [{
                "dcls_month": "202608",
                "fin_co_no": "0010001",
                "fin_prdt_cd": f"P{page}",
                "kor_co_nm": "합성은행",
                "fin_prdt_nm": "합성상품",
                "max_limit": None,
                "dcls_end_day": None,
            }],
            "optionList": [option],
        }
    }


def source(handler, **kwargs):
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return FinlifeHttpFinancialProductSource(
        api_key=SecretStr(SYNTHETIC_KEY), client=client, **kwargs
    )


def test_builds_endpoint_and_params_without_logging_secret(caplog):
    observed = {}

    def handler(request):
        observed["scheme"] = request.url.scheme
        observed["path"] = request.url.path
        observed["params"] = dict(request.url.params)
        return httpx.Response(200, json=payload(), request=request)

    with caplog.at_level(logging.DEBUG):
        result = source(handler).fetch(ProductType.SAVINGS)

    assert observed == {
        "scheme": "https",
        "path": "/finlifeapi/savingProductsSearch.json",
        "params": {
            "auth": SYNTHETIC_KEY,
            "topFinGrpNo": "020000",
            "pageNo": "1",
        },
    }
    assert result.product_type == ProductType.SAVINGS
    assert SYNTHETIC_KEY not in caplog.text
    assert "auth=" not in caplog.text


def test_missing_key_and_non_https_are_rejected_safely():
    missing = FinlifeHttpFinancialProductSource(api_key=None)
    with pytest.raises(FinlifeHttpSourceError) as error:
        missing.fetch(ProductType.SAVINGS)
    assert error.value.kind == FinlifeHttpErrorKind.MISSING_API_KEY
    assert "auth" not in str(error.value).lower()

    with pytest.raises(FinlifeHttpSourceError) as error:
        FinlifeHttpFinancialProductSource(
            api_key=SecretStr(SYNTHETIC_KEY), base_url="http://finlife.example"
        )
    assert error.value.kind == FinlifeHttpErrorKind.INVALID_CONFIGURATION


def test_settings_loads_optional_secret_from_environment_and_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("FINLIFE_API_KEY")
    without_key = Settings(OPENAI_API_KEY="synthetic-openai", _env_file=None)
    assert without_key.FINLIFE_API_KEY is None

    monkeypatch.setenv("FINLIFE_API_KEY", SYNTHETIC_KEY)
    from_environment = Settings(OPENAI_API_KEY="synthetic-openai", _env_file=None)
    assert from_environment.FINLIFE_API_KEY.get_secret_value() == SYNTHETIC_KEY
    assert SYNTHETIC_KEY not in repr(from_environment)

    monkeypatch.delenv("FINLIFE_API_KEY")
    env_file = tmp_path / "synthetic.env"
    env_file.write_text(f"FINLIFE_API_KEY={SYNTHETIC_KEY}\n", encoding="utf-8")
    from_file = Settings(OPENAI_API_KEY="synthetic-openai", _env_file=env_file)
    assert from_file.FINLIFE_API_KEY.get_secret_value() == SYNTHETIC_KEY
    assert SYNTHETIC_KEY not in repr(from_file)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, FinlifeHttpErrorKind.RATE_LIMITED),
        (404, FinlifeHttpErrorKind.HTTP_CLIENT_ERROR),
        (503, FinlifeHttpErrorKind.HTTP_SERVER_ERROR),
    ],
)
def test_classifies_http_status_without_body(status, expected):
    def handler(request):
        return httpx.Response(status, text=f"secret body {SYNTHETIC_KEY}", request=request)

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == expected
    assert SYNTHETIC_KEY not in str(error.value)
    assert "secret body" not in str(error.value)


@pytest.mark.parametrize(
    ("exception_type", "expected"),
    [
        (httpx.ConnectTimeout, FinlifeHttpErrorKind.CONNECT_TIMEOUT),
        (httpx.ReadTimeout, FinlifeHttpErrorKind.READ_TIMEOUT),
        (httpx.WriteTimeout, FinlifeHttpErrorKind.WRITE_TIMEOUT),
        (httpx.PoolTimeout, FinlifeHttpErrorKind.POOL_TIMEOUT),
        (httpx.ConnectError, FinlifeHttpErrorKind.CONNECTION_ERROR),
    ],
)
def test_classifies_transport_errors(exception_type, expected):
    def handler(request):
        raise exception_type(f"unsafe {request.url!s}", request=request)

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == expected
    assert SYNTHETIC_KEY not in str(error.value)
    assert "https://" not in str(error.value)


def test_classifies_tls_error():
    def handler(request):
        try:
            raise ssl.SSLCertVerificationError("synthetic certificate failure")
        except ssl.SSLError as exc:
            raise httpx.ConnectError("unsafe transport detail", request=request) from exc

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == FinlifeHttpErrorKind.TLS_ERROR


@pytest.mark.parametrize(
    "location",
    [
        "http://finlife.fss.or.kr/insecure",
        "https://other.example/redirect",
        "/same-host-but-not-followed",
    ],
)
def test_redirects_are_never_followed(location):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(302, headers={"location": location}, request=request)

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == FinlifeHttpErrorKind.REDIRECT_REJECTED
    assert calls == 1


def test_rejects_client_configured_to_follow_redirects():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        follow_redirects=True,
    )
    with pytest.raises(FinlifeHttpSourceError) as error:
        FinlifeHttpFinancialProductSource(
            api_key=SecretStr(SYNTHETIC_KEY), client=client
        )
    assert error.value.kind == FinlifeHttpErrorKind.INVALID_CONFIGURATION


def test_classifies_json_schema_and_finlife_errors():
    cases = [
        (lambda request: httpx.Response(200, text="not-json", request=request), FinlifeHttpErrorKind.INVALID_JSON),
        (lambda request: httpx.Response(200, json={"result": {}}, request=request), FinlifeHttpErrorKind.INVALID_RESPONSE_SCHEMA),
        (lambda request: httpx.Response(200, json={"result": {"err_cd": "020", "err_msg": "limit"}}, request=request), FinlifeHttpErrorKind.FINLIFE_API_ERROR),
    ]
    for handler, expected in cases:
        with pytest.raises(FinlifeHttpSourceError) as error:
            source(handler).fetch(ProductType.SAVINGS)
        assert error.value.kind == expected


def test_allows_missing_prdt_div_and_rejects_mismatch():
    without_div = payload()
    without_div["result"].pop("prdt_div")
    result = source(
        lambda request: httpx.Response(200, json=without_div, request=request)
    ).fetch(ProductType.SAVINGS)
    assert result.pages[0].product_type == ProductType.SAVINGS

    mismatch = payload()
    mismatch["result"]["prdt_div"] = "D"
    with pytest.raises(FinlifeHttpSourceError) as error:
        source(lambda request: httpx.Response(200, json=mismatch, request=request)).fetch(
            ProductType.SAVINGS
        )
    assert error.value.kind == FinlifeHttpErrorKind.INVALID_RESPONSE_SCHEMA


def test_collects_all_pages_and_validates_pagination():
    requested = []

    def handler(request):
        page_no = int(request.url.params["pageNo"])
        requested.append(page_no)
        return httpx.Response(
            200,
            json=payload(page=page_no, total=2, maximum=2),
            request=request,
        )

    result = source(handler).fetch(ProductType.SAVINGS)
    assert requested == [1, 2]
    assert result.completeness.is_complete
    assert len(result.pages) == 2


def test_page_limit_is_always_incomplete():
    result = source(
        lambda request: httpx.Response(200, json=payload(), request=request)
    ).fetch(ProductType.SAVINGS, page_limit=1)
    assert result.page_limited is True
    assert result.completeness.is_complete is False


def test_middle_page_failure_returns_no_snapshot():
    def handler(request):
        page_no = int(request.url.params["pageNo"])
        if page_no == 2:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json=payload(page=1, total=3, maximum=3), request=request)

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == FinlifeHttpErrorKind.HTTP_SERVER_ERROR


def test_detects_pagination_mismatch():
    def handler(request):
        page_no = int(request.url.params["pageNo"])
        response_payload = payload(page=page_no, total=2, maximum=2)
        if page_no == 2:
            response_payload["result"]["total_count"] = 3
        return httpx.Response(200, json=response_payload, request=request)

    with pytest.raises(FinlifeHttpSourceError) as error:
        source(handler).fetch(ProductType.SAVINGS)
    assert error.value.kind == FinlifeHttpErrorKind.INVALID_PAGINATION


def test_deposit_endpoint_determines_type_and_timeout_is_explicit():
    observed = {}

    def handler(request):
        observed["path"] = request.url.path
        observed["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200, json=payload(ProductType.DEPOSIT), request=request
        )

    result = source(
        handler,
        timeouts=FinlifeHttpTimeouts(connect=1, read=2, write=3, pool=4),
    ).fetch(ProductType.DEPOSIT)
    assert observed["path"] == "/finlifeapi/depositProductsSearch.json"
    assert observed["timeout"] == {"connect": 1, "read": 2, "write": 3, "pool": 4}
    assert result.product_type == ProductType.DEPOSIT


def test_inspection_contains_only_structure_and_counts():
    result = source(
        lambda request: httpx.Response(200, json=payload(), request=request)
    ).fetch(ProductType.SAVINGS, page_limit=1)
    serialized = json.dumps(result.inspections[0].__dict__, ensure_ascii=False)
    assert "합성상품" not in serialized
    assert "special_conditions" not in serialized
    assert SYNTHETIC_KEY not in serialized
    assert result.inspections[0].base_count == 1
    assert "max_limit" in result.inspections[0].base_fields_nullable
