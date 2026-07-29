from pydantic import BaseModel

class LLMTestRequest(BaseModel):
    """
    LLM 호출 테스트 요청 DTO
    """
    prompt: str

class LLMTestResponse(BaseModel):
    """
    LLM 호출 테스트 응답 DTO
    """
    result: str