from langchain_openai import ChatOpenAI
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )

    async def generate_test(self, prompt: str) -> str:
        response = await self.llm.ainvoke(prompt)
        return response.content

llm_service = LLMService()