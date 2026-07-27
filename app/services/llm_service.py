from langchain_openai import ChatOpenAI
from app.core.config import settings

class LLMService:
    """
    LLM 모델 호출 및 비동기 추론 로직을 전담하는 서비스 클래스
    """
    def __init__(self):
        # LangChain 기반 ChatOpenAI 모델 인스턴스 초기화(gpt-4o-mini 모델 적용)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )

    async def generate_test(self, prompt: str) -> str:
        """
        입력받은 프롬프트를 LLM에 비동기로 전달하고 생성된 응답 텍스트를 반환
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

# 전역에서 사용할 LLM 서비스 싱글톤 인스턴스 생성
llm_service = LLMService()