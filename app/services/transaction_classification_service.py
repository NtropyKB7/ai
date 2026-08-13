import re
from collections import Counter
from typing import Optional

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionClassificationResult,
    TransactionForClassification,
)


class TransactionClassificationService:
    """
    규칙 기반 분류를 먼저 적용하고,
    결정되지 않은 거래만 LLM으로 보조 분류합니다.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    async def classify_transactions_with_llm(
        self,
        transactions: list[TransactionForClassification],
    ) -> list[TransactionClassificationResult]:
        """
        FastAPI 라우터의 기존 비동기 호출 계약을 유지합니다.

        실제 규칙·LLM 분류와 fallback 처리는
        classify_transactions()에서 수행합니다.
        """
        return self.classify_transactions(transactions)

    def classify_transactions(
        self,
        transactions: list[TransactionForClassification],
    ) -> list[TransactionClassificationResult]:
        """
        모든 입력 transactionId에 대해 분류 결과를 하나씩 반환합니다.

        처리 순서:
        1. 규칙 기반 분류
        2. 미분류 거래 LLM 보조 분류
        3. LLM 결과 검증
        4. 최종 실패 거래 ETC / VARIABLE 처리
        """
        rule_results: dict[int, TransactionClassificationResult] = {}
        unclassified_transactions: list[TransactionForClassification] = []

        for transaction in transactions:
            rule_result = self._classify_by_rules(transaction)

            if rule_result is not None:
                rule_results[transaction.transactionId] = rule_result
            else:
                unclassified_transactions.append(transaction)

        llm_results: list[TransactionClassificationResult] = []

        if unclassified_transactions and self.llm_service:
            llm_results = self.llm_service.classify_with_llm(
                unclassified_transactions
            )

        unclassified_ids = {
            transaction.transactionId
            for transaction in unclassified_transactions
        }

        result_counts = Counter(
            result.transactionId
            for result in llm_results
            if result.transactionId in unclassified_ids
        )

        llm_result_map = {
            result.transactionId: result
            for result in llm_results
            if result.transactionId in unclassified_ids
            and result_counts[result.transactionId] == 1
            and self._is_valid_result(result)
        }

        final_results: list[TransactionClassificationResult] = []

        for transaction in transactions:
            transaction_id = transaction.transactionId

            if transaction_id in rule_results:
                final_results.append(rule_results[transaction_id])
                continue

            if transaction_id in llm_result_map:
                final_results.append(llm_result_map[transaction_id])
                continue

            final_results.append(
                self._create_fallback_result(transaction_id)
            )

        return final_results

    @staticmethod
    def _create_fallback_result(
        transaction_id: int,
    ) -> TransactionClassificationResult:
        """
        규칙과 LLM 모두 최종 결과를 만들지 못하면
        기타 변동 소비로 확정합니다.
        """
        return TransactionClassificationResult(
            transactionId=transaction_id,
            isConsumption=True,
            category=ExpenseCategory.ETC,
            expenseType=ExpenseType.VARIABLE,
        )

    @staticmethod
    def _is_valid_result(
        result: TransactionClassificationResult,
    ) -> bool:
        """
        소비 결과에는 category와 expenseType이 모두 있어야 하고,
        비소비 결과에는 두 값이 모두 없어야 합니다.
        """
        if result.isConsumption:
            return (
                result.category is not None
                and result.expenseType is not None
            )

        return (
            result.category is None
            and result.expenseType is None
        )

    def _classify_by_rules(
        self,
        transaction: TransactionForClassification,
    ) -> Optional[TransactionClassificationResult]:
        """
        CODEF 원본 설명 필드를 이용하여 규칙 기반 분류를 수행합니다.

        desc3은 대부분 은행에서 상대방·가맹점·상품명이므로 가장 먼저
        사용하고, desc1·desc2·desc4는 보조 정보로 사용합니다.
        """
        target_text = " ".join(
            value.strip()
            for value in (
                transaction.desc3,
                transaction.desc1,
                transaction.desc2,
                transaction.desc4,
            )
            if value and value.strip()
        )

        normalized_text = re.sub(r"\s+", "", target_text)

        # 금융 목적이 명확한 자산 이동만 비소비로 처리합니다.
        # 자동이체·CMS·FBS·전자금융 등의 거래 방식만으로는
        # 비소비라고 판단하지 않습니다.
        non_consumption_keywords = [
            "정기적금",
            "자유적금",
            "적금납입",
            "정기예금",
            "예금납입",
            "청약",
            "대출상환",
            "대출원금",
            "원금상환",
            "대출계좌",
            "수시입출금계좌",
            "본인계좌",
            "내계좌",
            "주식",
            "펀드",
            "투자",
        ]

        if self._contains_keyword(
            normalized_text,
            non_consumption_keywords,
        ):
            return self._create_non_consumption_result(
                transaction.transactionId
            )

        # 대출 이자
        loan_interest_keywords = [
            "대출이자",
            "이자납입",
        ]

        if self._contains_keyword(
            normalized_text,
            loan_interest_keywords,
        ):
            return self._create_consumption_result(
                transaction.transactionId,
                ExpenseCategory.FINANCE,
                ExpenseType.FIXED,
            )

        # 보험료
        insurance_keywords = [
            "보험",
            "생명",
            "화재",
            "손해보험",
            "해상보험",
        ]

        if self._contains_keyword(
            normalized_text,
            insurance_keywords,
        ):
            return self._create_consumption_result(
                transaction.transactionId,
                ExpenseCategory.INSURANCE,
                ExpenseType.FIXED,
            )

        # 신용카드 대금
        # 체크, BC, C/C 같은 결제수단 표시는 카드대금으로 판단하지 않습니다.
        card_bill_keywords = [
            "카드대금",
            "신용카드대금",
            "카드이용대금",
            "카드결제대금",
            "카드청구",
        ]

        if self._contains_keyword(
            normalized_text,
            card_bill_keywords,
        ):
            return self._create_consumption_result(
                transaction.transactionId,
                ExpenseCategory.ETC,
                ExpenseType.VARIABLE,
            )

        # 현금 인출
        cash_withdrawal_keywords = [
            "현금인출",
            "ATM출금",
            "ATM인출",
            "스마트출금",
            "CD출금",
            "자동화기기출금",
        ]

        if self._contains_keyword(
            normalized_text,
            cash_withdrawal_keywords,
        ):
            return self._create_consumption_result(
                transaction.transactionId,
                ExpenseCategory.ETC,
                ExpenseType.VARIABLE,
            )

        # 일회성 금융 수수료
        financial_fee_keywords = [
            "금융수수료",
            "이체수수료",
            "해외결제수수료",
        ]

        if self._contains_keyword(
            normalized_text,
            financial_fee_keywords,
        ):
            return self._create_consumption_result(
                transaction.transactionId,
                ExpenseCategory.FINANCE,
                ExpenseType.VARIABLE,
            )

        category_rules: list[
            tuple[str, ExpenseCategory, ExpenseType]
        ] = [
            (
                r"(배달의민족|요기요|쿠팡이츠|식당|음식점|카페|"
                r"스타벅스|투썸|할리스|이디야|베이커리|"
                r"한식|중식|일식|양식|치킨|피자)",
                ExpenseCategory.FOOD,
                ExpenseType.VARIABLE,
            ),
            (
                r"(시내버스|지하철|택시|카카오T|티머니|코레일|"
                r"KTX|SRT|주유소|SK에너지|GS칼텍스|"
                r"S-OIL|현대오일뱅크)",
                ExpenseCategory.TRANSPORTATION,
                ExpenseType.VARIABLE,
            ),
            (
                r"(월세|관리비|도시가스|전기요금|수도요금|한국전력)",
                ExpenseCategory.HOUSING,
                ExpenseType.FIXED,
            ),
            (
                r"(SKT|KT|LGU\+|알뜰폰|통신비|유선방송|인터넷요금)",
                ExpenseCategory.COMMUNICATION,
                ExpenseType.FIXED,
            ),
            (
                r"(병원|의원|약국|치과|한의원|안과|피부과|"
                r"정형외과|내과)",
                ExpenseCategory.MEDICAL,
                ExpenseType.VARIABLE,
            ),
            (
                r"(학원|등록금|독서실|스터디카페|인프런|"
                r"패스트캠퍼스|유데미|수강료)",
                ExpenseCategory.EDUCATION,
                ExpenseType.FIXED,
            ),
            (
                r"(쿠팡|11번가|네이버페이|G마켓|옥션|무신사|"
                r"백화점|마트|이마트|홈플러스|롯데마트|"
                r"다이소|올리브영)",
                ExpenseCategory.SHOPPING,
                ExpenseType.VARIABLE,
            ),
            (
                r"(CGV|메가박스|롯데시네마|넷플릭스|티빙|"
                r"웨이브|왓챠|멜론|지니|PC방|골프|헬스|"
                r"피트니스)",
                ExpenseCategory.LEISURE,
                ExpenseType.VARIABLE,
            ),
        ]

        for pattern, category, expense_type in category_rules:
            if re.search(pattern, target_text, re.IGNORECASE):
                return self._create_consumption_result(
                    transaction.transactionId,
                    category,
                    expense_type,
                )

        # 규칙으로 결정하지 못한 ORDINARY 거래만 LLM으로 전달됩니다.
        return None

    @staticmethod
    def _contains_keyword(
        normalized_text: str,
        keywords: list[str],
    ) -> bool:
        return any(
            re.sub(r"\s+", "", keyword) in normalized_text
            for keyword in keywords
        )

    @staticmethod
    def _create_consumption_result(
        transaction_id: int,
        category: ExpenseCategory,
        expense_type: ExpenseType,
    ) -> TransactionClassificationResult:
        return TransactionClassificationResult(
            transactionId=transaction_id,
            isConsumption=True,
            category=category,
            expenseType=expense_type,
        )

    @staticmethod
    def _create_non_consumption_result(
        transaction_id: int,
    ) -> TransactionClassificationResult:
        return TransactionClassificationResult(
            transactionId=transaction_id,
            isConsumption=False,
            category=None,
            expenseType=None,
        )