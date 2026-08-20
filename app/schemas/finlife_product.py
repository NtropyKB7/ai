from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


OptionalInt = Annotated[Optional[int], BeforeValidator(_optional_int)]


class ProductType(str, Enum):
    CARD = "CARD"
    SAVINGS = "SAVINGS"
    DEPOSIT = "DEPOSIT"


class FinlifeBaseInfo(BaseModel):
    dcls_month: str
    fin_co_no: str
    fin_prdt_cd: str
    kor_co_nm: Optional[str] = None
    fin_prdt_nm: Optional[str] = None
    join_way: Optional[str] = None
    mtrt_int: Optional[str] = None
    spcl_cnd: Optional[str] = None
    join_deny: Optional[str] = None
    join_member: Optional[str] = None
    etc_note: Optional[str] = None
    max_limit: OptionalInt = None
    dcls_strt_day: Optional[str] = None
    dcls_end_day: Optional[str] = None
    fin_co_subm_day: Optional[str] = None

    @property
    def join_key(self) -> tuple[str, str, str]:
        return self.dcls_month, self.fin_co_no, self.fin_prdt_cd


class FinlifeOptionBase(BaseModel):
    dcls_month: str
    fin_co_no: str
    fin_prdt_cd: str
    intr_rate_type: Optional[str] = None
    intr_rate_type_nm: Optional[str] = None
    save_trm: Optional[str] = None
    intr_rate: Optional[Decimal] = None
    intr_rate2: Optional[Decimal] = None

    @property
    def join_key(self) -> tuple[str, str, str]:
        return self.dcls_month, self.fin_co_no, self.fin_prdt_cd


class FinlifeSavingsOption(FinlifeOptionBase):
    rsrv_type: Optional[str] = None
    rsrv_type_nm: Optional[str] = None


class FinlifeDepositOption(FinlifeOptionBase):
    model_config = ConfigDict(extra="forbid")


class FinlifeResultBase(BaseModel):
    prdt_div: Optional[str] = None
    total_count: OptionalInt = None
    max_page_no: OptionalInt = None
    now_page_no: OptionalInt = None
    err_cd: str
    err_msg: str

    @property
    def is_success(self) -> bool:
        return self.err_cd == "000"


class FinlifeSavingsResult(FinlifeResultBase):
    baseList: Optional[list[FinlifeBaseInfo]] = None
    optionList: Optional[list[FinlifeSavingsOption]] = None


class FinlifeDepositResult(FinlifeResultBase):
    baseList: Optional[list[FinlifeBaseInfo]] = None
    optionList: Optional[list[FinlifeDepositOption]] = None


class FinlifeSavingsResponse(BaseModel):
    result: FinlifeSavingsResult


class FinlifeDepositResponse(BaseModel):
    result: FinlifeDepositResult


class SavingsOptionDetails(BaseModel):
    interest_rate_type: Optional[str] = None
    interest_rate_type_name: Optional[str] = None
    reserve_type: Optional[str] = None
    reserve_type_name: Optional[str] = None
    term_months: Optional[int] = None
    base_interest_rate: Optional[Decimal] = None
    preferred_interest_rate: Optional[Decimal] = None


class DepositOptionDetails(BaseModel):
    interest_rate_type: Optional[str] = None
    interest_rate_type_name: Optional[str] = None
    term_months: Optional[int] = None
    base_interest_rate: Optional[Decimal] = None
    preferred_interest_rate: Optional[Decimal] = None


class ExternalProductDetailsBase(BaseModel):
    source: Literal["FSS_FINLIFE"] = "FSS_FINLIFE"
    source_region_code: Literal["020000"] = "020000"
    source_product_id: str
    source_provider_id: str
    disclosure_month: str
    disclosure_start_date: Optional[str] = None
    disclosure_end_date: Optional[str] = None
    source_submitted_at: Optional[str] = None
    collected_at: datetime
    join_way: Optional[str] = None
    maturity_interest_text: Optional[str] = None
    special_conditions_text: Optional[str] = None
    join_restriction_code: Optional[str] = None
    eligible_members_text: Optional[str] = None
    notes_text: Optional[str] = None
    max_limit: Optional[int] = None


class SavingsProductDetails(ExternalProductDetailsBase):
    product_type: Literal[ProductType.SAVINGS] = ProductType.SAVINGS
    options: list[SavingsOptionDetails] = Field(default_factory=list)


class DepositProductDetails(ExternalProductDetailsBase):
    product_type: Literal[ProductType.DEPOSIT] = ProductType.DEPOSIT
    options: list[DepositOptionDetails] = Field(default_factory=list)


ExternalProductDetails = Union[SavingsProductDetails, DepositProductDetails]


class FinlifePage(BaseModel):
    top_fin_group_no: Literal["020000"] = "020000"
    product_type: ProductType
    total_count: int
    max_page_no: int
    now_page_no: int
    bases: list[FinlifeBaseInfo]
    options: list[Union[FinlifeSavingsOption, FinlifeDepositOption]]

    @model_validator(mode="after")
    def validate_product_type(self):
        if self.product_type not in (ProductType.SAVINGS, ProductType.DEPOSIT):
            raise ValueError("Finlife page supports SAVINGS or DEPOSIT only")
        return self


class SnapshotCompleteness(BaseModel):
    all_regions_succeeded: bool = True
    all_pages_succeeded: bool = True
    all_responses_successful: bool = True
    page_numbers_contiguous: bool = True
    pagination_consistent: bool = True
    non_empty_and_not_abnormal_drop: bool = True

    @property
    def is_complete(self) -> bool:
        return all(self.model_dump().values())
