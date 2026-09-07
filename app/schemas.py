from pydantic import BaseModel, Field, field_validator


class IndexResponse(BaseModel):
    files_indexed: int
    sections_indexed: int


class ChatRequest(BaseModel):
    query: str = Field(max_length=2000)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("query must be a string")

        value = value.strip()

        if not value:
            raise ValueError("query must not be empty")

        return value


class SourceInfo(BaseModel):
    source: str
    heading: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]