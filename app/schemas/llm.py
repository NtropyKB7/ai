from pydantic import BaseModel

class LLMTestRequest(BaseModel):
    prompt: str

class LLMTestResponse(BaseModel):
    result: str