from pydantic import BaseModel, Field

class QAResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Source paths or IDs")

