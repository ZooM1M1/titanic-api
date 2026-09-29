from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    sex: str = Field(...)
    age: float = Field(..., ge=0, le=120)
    fare: float = Field(..., ge=0)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    embarked: str = Field(...)
    title: str = Field(...)


class PredictionOutput(BaseModel):
    survived: int = Field(...)
    probability: float = Field(...)
    message: str = Field(...)


# Явная пересборка моделей — лечит ошибку "not fully defined" на Python 3.14
PassengerInput.model_rebuild()
PredictionOutput.model_rebuild()