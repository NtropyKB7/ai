import re
from typing import Optional

from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionClassificationResult,
    TransactionForClassification,
)


class TransactionClassificationService:
    """
    소비 내역 분류 비즈니스 로직을 처리하는 서비스 클래스입니다.

    규칙 기반(Rule-based) 분류를 우선 적용하고,
    규칙으로 분류되지 않는 거래는 LLM을 활용하여 보조 분류합니다.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def classify_transactions(
        self, transactions: list[TransactionForClassification]
    ) -> list[TransactionClassificationResult]:
        """
        일괄 거래 내역을 받아 각 거래별 소비 여부, 카테고리, 지출 유형을 분류합니다.

        1. 규칙 기반 분류 우선 수행
        2. 미분류 항목이 존재할 경우 LLM 서비스에 보조 분류 요청
        3. 모든 transactionId에 대해 1:1 분류 결과 응답 보장
        """
        classified_results: list[TransactionClassificationResult] = []
        unclassified_transactions: list[TransactionForClassification] = []

        # 1. 규칙 기반 1차 분류
        for txn in transactions:
            rule_result = self._classify_by_rules(txn)
            if rule_result:
                classified_results.append(rule_result)
            else:
                unclassified_transactions.append(txn)

        # 2. 미분류 건에 대한 LLM 보조 분류
        if unclassified_transactions and self.llm_service:
            llm_results = self.llm_service.classify_with_llm(unclassified_transactions)
            classified_results.extend(llm_results)

        # 3. 입력 순서 및 1:1 응답 매핑 보장
        result_map = {res.transactionId: res for res in classified_results}
        final_results: list[TransactionClassificationResult] = []

        for txn in transactions:
            if txn.transactionId in result_map:
                final_results.append(result_map[txn.transactionId])
            else:
                # 규칙 및 LLM 모두 실패 시 안전 기본값 (기타 변동 소비) 처리
                final_results.append(
                    TransactionClassificationResult(
                        transactionId=txn.transactionId,
                        isConsumption=True,
                        category=ExpenseCategory.ETC,
                        expenseType=ExpenseType.VARIABLE,
                    )
                )

        return final_results

    def _classify_by_rules(
        self, txn: TransactionForClassification
    ) -> Optional[TransactionClassificationResult]:
        """
        키워드 및 정규식을 활용하여 규칙 기반 분류를 수행합니다.
        """
        # 가맹점명과 상세 설명을 결합하여 검색 대상 텍스트 생성
        merchant = txn.merchantName or ""
        desc = txn.description or ""
        target_text = f"{merchant} {desc}".strip()

        # =========================================================================
        # 1. 비소비 거래 판단 (자산 이동, 저축, 원금 상환, 현금 인출)
        # =========================================================================
        non_consumption_keywords = [
            "적금",
            "예금",
            "청약",
            "대출원금",
            "원금상환",
            "본인이체",
            "내계좌",
            "계좌이체",
            "현금인출",
            "ATM인출",
            "주식",
            "펀드",
            "투자",
        ]
        if any(keyword in target_text for keyword in non_consumption_keywords):
            return TransactionClassificationResult(
                transactionId=txn.transactionId,
                isConsumption=False,
                category=None,
                expenseType=None,
            )

        # =========================================================================
        # 2. 금융 예외 소비 거래 판단
        # =========================================================================
        # 2-1. 대출 이자 (소비 / FINANCE / FIXED)
        if "대출이자" in target_text or "이자납입" in target_text:
            return TransactionClassificationResult(
                transactionId=txn.transactionId,
                isConsumption=True,
                category=ExpenseCategory.FINANCE,
                expenseType=ExpenseType.FIXED,
            )

        # 2-2. 보험료 (소비 / INSURANCE / FIXED)
        insurance_keywords = ["보험", "생명", "화재", "손해보험", "해상보험"]
        if any(keyword in target_text for keyword in insurance_keywords):
            return TransactionClassificationResult(
                transactionId=txn.transactionId,
                isConsumption=True,
                category=ExpenseCategory.INSURANCE,
                expenseType=ExpenseType.FIXED,
            )

        # 2-3. 카드대금 납부 (MVP 예외: 소비 / ETC / VARIABLE)
        card_keywords = ["카드대금", "카드결제", "카드이용대금", "카드청구"]
        if any(keyword in target_text for keyword in card_keywords):
            return TransactionClassificationResult(
                transactionId=txn.transactionId,
                isConsumption=True,
                category=ExpenseCategory.ETC,
                expenseType=ExpenseType.VARIABLE,
            )

        # =========================================================================
        # 3. 카테고리별 패턴 규칙 매칭
        # =========================================================================
        category_rules: list[tuple[str, ExpenseCategory, ExpenseType]] = [
            # 식비
            (
                r"(배달의민족|요기요|쿠팡이츠|식당|음식점|카페|스타벅스|투썸|할리스|이디야|베이커리|한식|중식|일식|양식|치킨|피자)",
                ExpenseCategory.FOOD,
                ExpenseType.VARIABLE,
            ),
            # 교통·주유
            (
                r"(시내버스|지하철|택시|카카오T|티머니|코레일|KTX|SRT|주유소|SK에너지|GS칼텍스|S-OIL|현대오일뱅크)",
                ExpenseCategory.TRANSPORTATION,
                ExpenseType.VARIABLE,
            ),
            # 주거
            (
                r"(월세|관리비|도시가스|전기요금|수도요금|한국전력)",
                ExpenseCategory.HOUSING,
                ExpenseType.FIXED,
            ),
            # 통신
            (
                r"(SKT|KT|LGU\+|알뜰폰|통신비|유선방송|인터넷요금)",
                ExpenseCategory.COMMUNICATION,
                ExpenseType.FIXED,
            ),
            # 의료·건강
            (
                r"(병원|의원|약국|치과|한의원|안과|피부과|정형외과|내과)",
                ExpenseCategory.MEDICAL,
                ExpenseType.VARIABLE,
            ),
            # 교육
            (
                r"(학원|등록금|독서실|스터디카페|인프런|패스트캠퍼스|유데미|수강료)",
                ExpenseCategory.EDUCATION,
                ExpenseType.FIXED,
            ),
            # 쇼핑
            (
                r"(쿠팡|11번가|네이버페이|G마켓|옥션|무신사|백화점|마트|이마트|홈플러스|롯데마트|다이소|올리브영)",
                ExpenseCategory.SHOPPING,
                ExpenseType.VARIABLE,
            ),
            # 취미·여가
            (
                r"(CGV|메가박스|롯데시네마|넷플릭스|티빙|웨이브|왓챠|멜론|지니|PC방|골프|헬스|피트니스)",
                ExpenseCategory.LEISURE,
                ExpenseType.VARIABLE,
            ),
        ]

        for pattern, category, expense_type in category_rules:
            if re.search(pattern, target_text, re.IGNORECASE):
                return TransactionClassificationResult(
                    transactionId=txn.transactionId,
                    isConsumption=True,
                    category=category,
                    expenseType=expense_type,
                )

        # 규칙 기반 분류 불가 시 None 반환 -> LLM 서비스로 전달
        return None