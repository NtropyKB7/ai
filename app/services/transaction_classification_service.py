# app/services/transaction_classification_service.py
import json
import logging

# LLM 호출 또는 JSON 검증 실패 상황을 기록하기 위한 logger입니다.
logger = logging.getLogger(__name__)

from pydantic import ValidationError

# DTO와 Enum은 #10에서 만든 transaction.py 파일에서 가져옵니다.
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionClassificationResult,
    TransactionForClassification,
    TransactionType,
    LLMTransactionClassificationResponse,
)


class TransactionClassificationService:

    

    """
    거래 내역을 규칙 기반으로 분류하는 서비스입니다.

    이 서비스는 DB에 저장하거나 FastAPI 응답 형식을 만들지 않습니다.
    오직 '거래 1건을 받아 분류 결과 1건을 만드는 일'만 담당합니다.
    """

    # 이 값보다 confidence가 낮으면 LLM 보조 분류 대상으로 봅니다.
    LOW_CONFIDENCE_THRESHOLD = 0.70
    
    # 소비가 아닌 거래를 판단하는 키워드입니다.
    # 입금(IN)은 아래 키워드를 확인하기 전에 바로 비소비로 처리합니다.
    NON_CONSUMPTION_KEYWORDS = (
        "계좌이체",
        "타행이체",
        "대출",
        "원금",
        "상환",
        "급여",
        "월급",
        "이자",
        "환급",
        "환불",
        "예금",
        "적금",
        "증권",
    )

    # 카테고리별 대표 키워드입니다.
    # 위에 작성된 카테고리부터 순서대로 검사하므로,
    # 더 구체적인 키워드가 있는 카테고리를 앞쪽에 둡니다.
    CATEGORY_KEYWORDS = {
        ExpenseCategory.FOOD: (
            "스타벅스",
            "커피",
            "카페",
            "식당",
            "음식",
            "배달",
            "쿠팡이츠",
            "배달의민족",
            "요기요",
            "편의점",
            "마트",
            "이마트",
            "홈플러스",
            "gs25",
            "cu",
        ),
        ExpenseCategory.TRANSPORT: (
            "버스",
            "지하철",
            "택시",
            "카카오t",
            "주유",
            "gs칼텍스",
            "sk에너지",
            "s-oil",
            "주차",
            "톨게이트",
        ),
        ExpenseCategory.HOUSING: (
            "관리비",
            "월세",
            "전세",
            "전기요금",
            "수도요금",
            "도시가스",
            "통신요금",
            "skt",
            "kt",
            "lg유플러스",
        ),
        ExpenseCategory.SUBSCRIPTION: (
            "넷플릭스",
            "유튜브",
            "멜론",
            "스포티파이",
            "디즈니",
            "티빙",
            "웨이브",
            "왓챠",
            "정기결제",
        ),
        ExpenseCategory.HEALTH: (
            "병원",
            "의원",
            "약국",
            "치과",
            "헬스",
            "건강검진",
        ),
        ExpenseCategory.EDUCATION: (
            "학원",
            "교육",
            "교재",
            "인강",
            "클래스",
            "udemy",
            "패스트캠퍼스",
        ),
        ExpenseCategory.FINANCE: (
            "보험",
            "카드수수료",
            "은행수수료",
            "증권수수료",
            "수수료",
        ),
        ExpenseCategory.LEISURE: (
            "영화",
            "극장",
            "공연",
            "여행",
            "숙박",
            "호텔",
            "놀이공원",
            "게임",
        ),
        ExpenseCategory.SHOPPING: (
            "쇼핑",
            "쿠팡",
            "백화점",
            "올리브영",
            "다이소",
            "무신사",
            "지마켓",
            "11번가",
        ),
    }

    # 반복적으로 빠져나가는 고정 지출을 판단하는 키워드입니다.
    FIXED_EXPENSE_KEYWORDS = (
        "관리비",
        "월세",
        "전세",
        "전기요금",
        "수도요금",
        "도시가스",
        "통신요금",
        "보험료",
        "정기결제",
        "자동납부",
    )

    def classify_transactions(
        self,
        transactions: list[TransactionForClassification],
    ) -> list[TransactionClassificationResult]:
        """
        여러 거래를 한 번에 분류합니다.

        FastAPI 라우터는 나중에 이 메서드 하나만 호출하면 됩니다.
        """

        return [
            self.classify_transaction(transaction)
            for transaction in transactions
        ]

    async def classify_transactions_with_llm(
        self,
        transactions: list[TransactionForClassification],
        llm_client=None,
    ) -> list[TransactionClassificationResult]:
        """
        규칙 기반 분류 후, 불확실한 거래만 LLM으로 보완 분류합니다.

        llm_client는 실제 운영에서는 전달하지 않습니다.
        테스트에서는 가짜 LLM 객체를 넣어 외부 API 호출 없이 검증할 수 있습니다.
        """

        # 1. 우선 모든 거래를 기존 규칙으로 분류합니다.
        rule_results = self.classify_transactions(transactions)

        # 2. OTHER이거나 낮은 confidence를 가진 결과만 LLM 대상으로 선별합니다.
        target_transaction_ids = {
            result.transactionId
            for result in rule_results
            if self._is_llm_target(result)
        }

        # LLM 보조 분류가 필요한 거래가 없다면 규칙 결과를 그대로 반환합니다.
        if not target_transaction_ids:
            return rule_results

        # 원본 거래 목록에서 LLM 대상 거래만 추립니다.
        target_transactions = [
            transaction
            for transaction in transactions
            if transaction.transactionId in target_transaction_ids
        ]

        try:
            # 실제 운영 환경에서는 이 시점에만 LLM 서비스를 import합니다.
            # 그래서 규칙 기반 단위 테스트는 OPENAI_API_KEY 없이도 실행할 수 있습니다.
            if llm_client is None:
                from app.services.llm_service import llm_service

                llm_client = llm_service

            # 3. 저신뢰도 거래들을 한 번에 LLM으로 전달합니다.
            raw_llm_response = await llm_client.classify_low_confidence_transactions(
                target_transactions,
            )

            # 4. LLM이 반환한 JSON을 파싱하고 DTO로 검증합니다.
            llm_results = self._parse_llm_results(raw_llm_response)

            # 5. 검증된 LLM 결과와 기존 규칙 결과를 병합합니다.
            return self._merge_llm_results(
                rule_results=rule_results,
                llm_results=llm_results,
                target_transaction_ids=target_transaction_ids,
            )

        except Exception as error:
            # LLM 호출 실패, JSON 파싱 실패, DTO 검증 실패가 발생해도
            # 전체 API를 실패시키지 않고 기존 규칙 결과를 반환합니다.
            logger.warning(
                "LLM 거래 보조 분류에 실패하여 규칙 기반 결과를 반환합니다: %s",
                error,
            )

            return rule_results

    def classify_transaction(
        self,
        transaction: TransactionForClassification,
    ) -> TransactionClassificationResult:
        """
        거래 1건을 규칙 기반으로 분류합니다.

        판단 순서:
        1. 입금 거래인지 확인
        2. 이체·대출·상환 등 비소비 키워드 확인
        3. 소비 카테고리 키워드 확인
        4. 분류하지 못한 출금은 OTHER / VARIABLE 처리
        """

        # 입금은 기본적으로 소비가 아니라고 판단합니다.
        if transaction.txnType == TransactionType.IN:
            return self._create_non_consumption_result(
                transaction_id=transaction.transactionId,
                confidence=0.99,
            )

        # 거래처명과 거래내용을 하나의 분류용 문자열로 만듭니다.
        classification_text = self._build_classification_text(transaction)

        # 출금이라도 계좌이체·대출 상환 등은 소비가 아닙니다.
        if self._contains_keyword(
            classification_text,
            self.NON_CONSUMPTION_KEYWORDS,
        ):
            return self._create_non_consumption_result(
                transaction_id=transaction.transactionId,
                confidence=0.95,
            )

        # 소비 카테고리를 키워드로 찾습니다.
        category = self._find_category(classification_text)

        # 어떤 카테고리에도 해당하지 않는 출금 거래입니다.
        if category is None:
            return TransactionClassificationResult(
                transactionId=transaction.transactionId,
                isConsumption=True,
                category=ExpenseCategory.OTHER,
                expenseType=ExpenseType.VARIABLE,
                confidence=0.40,
            )

        # 정기구독 또는 고정비 키워드가 있으면 FIXED입니다.
        expense_type = self._find_expense_type(
            category=category,
            classification_text=classification_text,
        )

        # 키워드가 명확히 일치한 소비 거래입니다.
        return TransactionClassificationResult(
            transactionId=transaction.transactionId,
            isConsumption=True,
            category=category,
            expenseType=expense_type,
            confidence=0.95,
        )

    def _build_classification_text(
        self,
        transaction: TransactionForClassification,
    ) -> str:
        """
        거래처명과 거래내용을 합쳐 분류에 사용할 문자열을 만듭니다.

        null 값은 제외하고, 대소문자와 공백 차이로
        키워드 분류가 실패하지 않도록 소문자·공백 제거를 적용합니다.
        """

        texts = [
            transaction.merchantName or "",
            transaction.transactionDetails or "",
        ]

        return " ".join(texts).lower().replace(" ", "")

    def _find_category(self, classification_text: str) -> ExpenseCategory | None:
        """분류용 문자열에서 가장 먼저 일치하는 소비 카테고리를 반환합니다."""

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if self._contains_keyword(classification_text, keywords):
                return category

        return None

    def _find_expense_type(
        self,
        category: ExpenseCategory,
        classification_text: str,
    ) -> ExpenseType:
        """
        고정/변동 지출을 판단합니다.

        정기구독은 거래처가 바뀌어도 반복성 지출로 보기 때문에
        키워드와 무관하게 FIXED로 처리합니다.
        """

        if category == ExpenseCategory.SUBSCRIPTION:
            return ExpenseType.FIXED

        if self._contains_keyword(
            classification_text,
            self.FIXED_EXPENSE_KEYWORDS,
        ):
            return ExpenseType.FIXED

        return ExpenseType.VARIABLE

    @staticmethod
    def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        """문자열에 키워드 중 하나라도 포함되면 True를 반환합니다."""

        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _create_non_consumption_result(
        transaction_id: int,
        confidence: float,
    ) -> TransactionClassificationResult:
        """
        비소비 거래 결과를 만드는 공통 메서드입니다.

        비소비 거래는 계약서 규칙대로 category와 expenseType을 null로 둡니다.
        """

        return TransactionClassificationResult(
            transactionId=transaction_id,
            isConsumption=False,
            category=None,
            expenseType=None,
            confidence=confidence,
        )


    def _is_llm_target(
        self,
        result: TransactionClassificationResult,
    ) -> bool:
        """
        LLM 보조 분류가 필요한 거래인지 판단합니다.

        OTHER 카테고리이거나 confidence가 기준값보다 낮으면 대상입니다.
        """

        return (
            result.category == ExpenseCategory.OTHER
            or result.confidence < self.LOW_CONFIDENCE_THRESHOLD
        )

    def _parse_llm_results(
        self,
        raw_llm_response: str,
    ) -> list[TransactionClassificationResult]:
        """
        LLM 원본 응답을 JSON으로 변환하고 DTO로 검증합니다.

        허용되지 않은 category, expenseType, confidence 범위는
        Pydantic ValidationError가 발생하여 fallback 처리됩니다.
        """

        # LLM이 실수로 ```json 코드 블록을 포함한 경우를 대비해 제거합니다.
        cleaned_response = self._remove_markdown_code_block(raw_llm_response)

        # JSON 문자열을 Python 딕셔너리로 변환합니다.
        response_data = json.loads(cleaned_response)

        # LLM 응답 구조와 Enum 값을 검증합니다.
        validated_response = LLMTransactionClassificationResponse.model_validate(
            response_data,
        )

        return validated_response.results

    @staticmethod
    def _remove_markdown_code_block(raw_response: str) -> str:
        """
        ```json ... ``` 형태로 응답한 경우 코드 블록 표시만 제거합니다.
        """

        response = raw_response.strip()

        if not response.startswith("```"):
            return response

        lines = response.splitlines()

        # 첫 번째 줄의 ``` 또는 ```json을 제거합니다.
        lines = lines[1:]

        # 마지막 줄이 ```이면 제거합니다.
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def _merge_llm_results(
        self,
        rule_results: list[TransactionClassificationResult],
        llm_results: list[TransactionClassificationResult],
        target_transaction_ids: set[int],
    ) -> list[TransactionClassificationResult]:
        """
        LLM이 정상적으로 반환한 대상 거래만 규칙 결과 대신 사용합니다.

        - 대상이 아닌 거래: 기존 규칙 결과 유지
        - LLM이 누락한 거래: 기존 규칙 결과 유지
        - LLM이 모르는 transactionId를 반환: 무시
        - LLM 결과의 소비/카테고리 관계가 이상함: 무시
        """

        # 대상 거래이고, 의미적으로도 올바른 LLM 결과만 저장합니다.
        valid_llm_results = {
            result.transactionId: result
            for result in llm_results
            if result.transactionId in target_transaction_ids
            and self._is_semantically_valid_llm_result(result)
        }

        # 기존 거래 순서가 유지되도록 결과를 병합합니다.
        return [
            valid_llm_results.get(rule_result.transactionId, rule_result)
            for rule_result in rule_results
        ]

    @staticmethod
    def _is_semantically_valid_llm_result(
        result: TransactionClassificationResult,
    ) -> bool:
        """
        Enum 검증만으로 확인할 수 없는 소비 여부와 null 관계를 확인합니다.
        """

        # 비소비라면 category와 expenseType은 둘 다 null이어야 합니다.
        if not result.isConsumption:
            return (
                result.category is None
                and result.expenseType is None
            )

        # 소비라면 category와 expenseType이 둘 다 있어야 합니다.
        return (
            result.category is not None
            and result.expenseType is not None
        )