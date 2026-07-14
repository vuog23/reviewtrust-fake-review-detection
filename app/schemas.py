from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12_000)
    classifier_threshold: float = Field(default=0.90, ge=0.50, le=0.99)
    ner_threshold: float = Field(default=0.50, ge=0.05, le=0.95)
    temperature: float = Field(default=0.70, ge=0.10, le=1.50)

    @field_validator("text")
    @classmethod
    def strip_and_reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review text must not be blank.")
        return value
