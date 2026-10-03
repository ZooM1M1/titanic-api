"""
FastAPI-сервер для предсказания выживания на Титанике.
Запуск: uvicorn main:app --reload
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from schemas import PassengerInput, PredictionOutput
# 1. Загружаем единый монолитный Pipeline при старте сервера
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model_pipeline.joblib"))
# 2. Создаём приложение
app = FastAPI(
    title="Titanic Survival API",
    description="Predicts survival of a Titanic passenger using Scikit-Learn Pipeline",
    version="2.0",
)
# 3. Превращаем входной объект Pydantic в датафрейм для пайплайна
def prepare_features(passenger: PassengerInput) -> pd.DataFrame:
    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = 1 if family_size == 1 else 0
    row = {
        "Pclass": passenger.pclass,
        "Sex": passenger.sex,
        "Age": passenger.age,
        "Fare": passenger.fare,
        "Embarked": passenger.embarked,
        "Title": passenger.title,
        "FamilySize": family_size,
        "IsAlone": is_alone,
    }
    return pd.DataFrame([row])
# 4. Корневой эндпоинт
@app.get("/")
def root():
    return {"message": "Titanic Survival API v2.0 (Pipeline). Go to /docs to test."}
# 5. Основной эндпоинт — предсказание
@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    X = prepare_features(passenger)
    # Пайплайн сам делает One-Hot, скейлинг, заполнение пропусков и предикт!
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0, 1])
    if pred == 1:
        msg = f"Passenger would survive with probability {proba:.1%}"
    else:
        msg = f"Passenger would die with probability {1 - proba:.1%}"
    return PredictionOutput(
        survived=pred,
        probability=proba,
        message=msg,
    )