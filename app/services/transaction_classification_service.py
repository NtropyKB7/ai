# app/services/transaction_classification_service.py

# DTO와 Enum은 #10에서 만든 transaction.py 파일에서 가져옵니다.
from app.schemas.transaction import (
    ExpenseCategory,
    ExpenseType,
    TransactionClassificationResult,
    TransactionForClassification,
    TransactionType,
)


class TransactionClassificationService:
    """
    거래 내역을 규칙 기반으로 분류하는 서비스입니다.

    이 서비스는 DB에 저장하거나 FastAPI 응답 형식을 만들지 않습니다.
    오직 '거래 1건을 받아 분류 결과 1건을 만드는 일'만 담당합니다.
    """

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