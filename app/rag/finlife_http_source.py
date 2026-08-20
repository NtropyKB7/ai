from __future__ import annotations

import json
import logging
import re
import ssl
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import SecretStr, ValidationError

from app.rag.financial_product_source import FinancialProductSource
from app.rag.finlife_response_parser import FinlifeResponseError, parse_finlife_response
from app.rag.snapshot_validator import validate_complete_pages
from app.schemas.finlife_product import FinlifePage, ProductType, SnapshotCompleteness


class _QueryStringRedactionFilter(logging.Filter):
    """Remove query strings emitted by httpx/httpcore before log formatting."""

    _query = re.compile(r"\?[^\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        record.msg = self._query.sub("?<redacted>", message)
        record.args = ()
        return True


def _install_http_log_redaction() -> None:
    for logger_name in ("httpx", "httpcore"):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(item, _QueryStringRedactionFilter) for item in logger.filters):
            logger.addFilter(_QueryStringRedactionFilter())


_install_http_log_redaction()


class FinlifeHttpErrorKind(str, Enum):
    RATE_LIMITED = "RATE_LIMITED"
    HTTP_CLIENT_ERROR = "HTTP_CLIENT_ERROR"
    HTTP_SERVER_ERROR = "HTTP_SERVER_ERROR"
    CONNECT_TIMEOUT = "CONNECT_TIMEOUT"
    READ_TIMEOUT = "READ_TIMEOUT"
    WRITE_TIMEOUT = "WRITE_TIMEOUT"
    POOL_TIMEOUT = "POOL_TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    TLS_ERROR = "TLS_ERROR"
    REDIRECT_REJECTED = "REDIRECT_REJECTED"
    INVALID_JSON = "INVALID_JSON"
    INVALID_RESPONSE_SCHEMA = "INVALID_RESPONSE_SCHEMA"
    FINLIFE_API_ERROR = "FINLIFE_API_ERROR"
    INVALID_PAGINATION = "INVALID_PAGINATION"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"


class FinlifeHttpSourceError(RuntimeError):
    """Safe public error that never copies the underlying request or response."""

    def __init__(
        self,
        kind: FinlifeHttpErrorKind,
        *,
        status_code: int | None = None,
        finlife_error_code: str | None = None,
    ):
        self.kind = kind
        self.status_code = status_code
        self.finlife_error_code = finlife_error_code
        super().__init__(f"Finlife HTTP collection failed ({kind.value})")


@dataclass(frozen=True)
class FinlifeHttpTimeouts:
    connect: float = 5.0
    read: float = 15.0
    write: float = 5.0
    pool: float = 5.0

    def as_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect,
            read=self.read,
            write=self.write,
            pool=self.pool,
        )


@dataclass(frozen=True)
class FinlifePageInspection:
    page_no: int
    http_status: int
    result_wrapper_present: bool
    base_list_present: bool
    option_list_present: bool
    prdt_div_present: bool
    err_cd: str | None
    total_count: int
    max_page_no: int
    now_page_no: int
    base_count: int
    option_count: int
    base_fields_present: tuple[str, ...] = field(default_factory=tuple)
    base_fields_nullable: tuple[str, ...] = field(default_factory=tuple)
    option_fields_present: tuple[str, ...] = field(default_factory=tuple)
    option_fields_nullable: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FinlifeFetchResult:
    product_type: ProductType
    endpoint_host: str
    endpoint_path: str
    pages: list[FinlifePage]
    completeness: SnapshotCompleteness
    inspections: list[FinlifePageInspection]
    page_limited: bool


class FinlifeHttpFinancialProductSource(FinancialProductSource):
    DEPOSIT_ALTERNATE_PATH = "/finlife/fdrmDpstApi/list.json"
    DEFAULT_PATHS = {
        ProductType.SAVINGS: "/finlifeapi/savingProductsSearch.json",
        ProductType.DEPOSIT: "/finlifeapi/depositProductsSearch.json",
    }

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        base_url: str = "https://finlife.fss.or.kr",
        endpoint_paths: dict[ProductType, str] | None = None,
        timeouts: FinlifeHttpTimeouts | None = None,
        client: httpx.Client | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") + "/"
        self._endpoint_paths = dict(self.DEFAULT_PATHS)
        if endpoint_paths:
            self._endpoint_paths.update(endpoint_paths)
        self._timeouts = timeouts or FinlifeHttpTimeouts()
        self._validate_configuration()
        if client is not None and client.follow_redirects:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
        self._client = client

    def fetch_pages(
        self, product_type: ProductType, top_fin_group_no: str = "020000"
    ) -> list[FinlifePage]:
        result = self.fetch(product_type, top_fin_group_no=top_fin_group_no)
        if not result.completeness.is_complete:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_PAGINATION)
        return result.pages

    def fetch(
        self,
        product_type: ProductType,
        *,
        top_fin_group_no: str = "020000",
        page_limit: int | None = None,
    ) -> FinlifeFetchResult:
        if product_type not in (ProductType.SAVINGS, ProductType.DEPOSIT):
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
        if top_fin_group_no != "020000" or page_limit is not None and page_limit < 1:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
        api_key = self._require_api_key()
        endpoint = self._endpoint(product_type)
        owned_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self._timeouts.as_httpx(),
            follow_redirects=False,
        )
        try:
            first_page, first_inspection = self._request_page(
                client, endpoint, product_type, top_fin_group_no, 1, api_key
            )
            if first_page.now_page_no != 1 or first_page.max_page_no < 1:
                raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_PAGINATION)
            requested_last_page = first_page.max_page_no
            if page_limit is not None:
                requested_last_page = min(requested_last_page, page_limit)
            pages = [first_page]
            inspections = [first_inspection]
            for page_no in range(2, requested_last_page + 1):
                page, inspection = self._request_page(
                    client,
                    endpoint,
                    product_type,
                    top_fin_group_no,
                    page_no,
                    api_key,
                )
                if (
                    page.now_page_no != page_no
                    or page.total_count != first_page.total_count
                    or page.max_page_no != first_page.max_page_no
                ):
                    raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_PAGINATION)
                pages.append(page)
                inspections.append(inspection)
            # An explicit page limit is an inspection run, never a writable snapshot.
            limited = page_limit is not None
            completeness = validate_complete_pages(pages)
            if limited:
                completeness = completeness.model_copy(
                    update={"all_pages_succeeded": False, "page_numbers_contiguous": False}
                )
            elif not completeness.is_complete:
                raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_PAGINATION)
            parsed = urlparse(endpoint)
            return FinlifeFetchResult(
                product_type=product_type,
                endpoint_host=parsed.hostname or "",
                endpoint_path=parsed.path,
                pages=pages,
                completeness=completeness,
                inspections=inspections,
                page_limited=limited,
            )
        finally:
            if owned_client:
                client.close()

    def _request_page(self, client, endpoint, product_type, group_no, page_no, api_key):
        try:
            response = client.get(
                endpoint,
                params={
                    "auth": api_key,
                    "topFinGrpNo": group_no,
                    "pageNo": str(page_no),
                },
                timeout=self._timeouts.as_httpx(),
            )
        except httpx.ConnectTimeout:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.CONNECT_TIMEOUT) from None
        except httpx.ReadTimeout:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.READ_TIMEOUT) from None
        except httpx.WriteTimeout:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.WRITE_TIMEOUT) from None
        except httpx.PoolTimeout:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.POOL_TIMEOUT) from None
        except httpx.ConnectError as exc:
            kind = (
                FinlifeHttpErrorKind.TLS_ERROR
                if self._has_ssl_cause(exc)
                else FinlifeHttpErrorKind.CONNECTION_ERROR
            )
            raise FinlifeHttpSourceError(kind) from None
        except httpx.TransportError:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.CONNECTION_ERROR) from None

        if 300 <= response.status_code < 400:
            self._validate_redirect(endpoint, response.headers.get("location"))
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.REDIRECT_REJECTED,
                status_code=response.status_code,
            )
        if response.status_code == 429:
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.RATE_LIMITED, status_code=429
            )
        if 400 <= response.status_code < 500:
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.HTTP_CLIENT_ERROR,
                status_code=response.status_code,
            )
        if response.status_code >= 500:
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.HTTP_SERVER_ERROR,
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_JSON) from None
        try:
            page = parse_finlife_response(payload, product_type)
        except FinlifeResponseError as exc:
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.FINLIFE_API_ERROR,
                finlife_error_code=exc.code,
            ) from None
        except (ValidationError, TypeError, ValueError, KeyError):
            raise FinlifeHttpSourceError(
                FinlifeHttpErrorKind.INVALID_RESPONSE_SCHEMA
            ) from None
        return page, self._inspect(payload, page, response.status_code)

    def _require_api_key(self) -> str:
        if self._api_key is None or not self._api_key.get_secret_value():
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.MISSING_API_KEY)
        return self._api_key.get_secret_value()

    def _endpoint(self, product_type: ProductType) -> str:
        endpoint = urljoin(self._base_url, self._endpoint_paths[product_type].lstrip("/"))
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
        return endpoint

    def _validate_configuration(self):
        parsed = urlparse(self._base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
        for product_type in (ProductType.SAVINGS, ProductType.DEPOSIT):
            if product_type not in self._endpoint_paths:
                raise FinlifeHttpSourceError(FinlifeHttpErrorKind.INVALID_CONFIGURATION)
            self._endpoint(product_type)

    @staticmethod
    def _has_ssl_cause(exc: BaseException) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, (ssl.SSLError, ssl.CertificateError)):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _validate_redirect(source: str, location: str | None):
        if not location:
            return
        target = urlparse(urljoin(source, location))
        origin = urlparse(source)
        if target.scheme != "https" or target.hostname != origin.hostname:
            raise FinlifeHttpSourceError(FinlifeHttpErrorKind.REDIRECT_REJECTED)

    @staticmethod
    def _inspect(payload: dict[str, Any], page: FinlifePage, status: int):
        result = payload.get("result") if isinstance(payload, dict) else None
        result = result if isinstance(result, dict) else {}
        bases = result.get("baseList") if isinstance(result.get("baseList"), list) else []
        options = result.get("optionList") if isinstance(result.get("optionList"), list) else []
        return FinlifePageInspection(
            page_no=page.now_page_no,
            http_status=status,
            result_wrapper_present="result" in payload,
            base_list_present="baseList" in result,
            option_list_present="optionList" in result,
            prdt_div_present="prdt_div" in result,
            err_cd=str(result["err_cd"]) if "err_cd" in result else None,
            total_count=page.total_count,
            max_page_no=page.max_page_no,
            now_page_no=page.now_page_no,
            base_count=len(bases),
            option_count=len(options),
            base_fields_present=tuple(sorted({key for row in bases for key in row})),
            base_fields_nullable=tuple(
                sorted({key for row in bases for key, value in row.items() if value is None})
            ),
            option_fields_present=tuple(sorted({key for row in options for key in row})),
            option_fields_nullable=tuple(
                sorted({key for row in options for key, value in row.items() if value is None})
            ),
        )
