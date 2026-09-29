"""
FastAPI-сервер для предсказания выживания на Титанике.
Запуск: uvicorn main:app --reload
"""

import pandas as pd
import joblib

from fastapi import FastAPI
from schemas import PassengerInput, PredictionOutput


import os

# 1. Загружаем модель и колонки при старте
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
columns = joblib.load(os.path.join(BASE_DIR, "columns.joblib"))


# 2. Создаём приложение
app = FastAPI(
    title="Titanic Survival API",
    description="Predicts survival of a Titanic passenger",
    version="1.0",
)


# 3. Превращаем вход в формат модели
def prepare_features(passenger: PassengerInput) -> pd.DataFrame:
    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = 1 if family_size == 1 else 0

    row = {
        "Pclass": passenger.pclass,
        "Age": passenger.age,
        "Fare": passenger.fare,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Sex_female": 1 if passenger.sex == "female" else 0,
        "Sex_male":   1 if passenger.sex == "male" else 0,
        "Embarked_C": 1 if passenger.embarked == "C" else 0,
        "Embarked_Q": 1 if passenger.embarked == "Q" else 0,
        "Embarked_S": 1 if passenger.embarked == "S" else 0,
        "Title_Master": 1 if passenger.title == "Master" else 0,
        "Title_Miss":   1 if passenger.title == "Miss" else 0,
        "Title_Mr":     1 if passenger.title == "Mr" else 0,
        "Title_Mrs":    1 if passenger.title == "Mrs" else 0,
        "Title_Rare":   1 if passenger.title == "Rare" else 0,
    }

    return pd.DataFrame([row])[columns]


# 4. Корневой эндпоинт
@app.get("/")
def root():
    return {"message": "Titanic Survival API. Go to /docs to test."}


# 5. Основной эндпоинт — предсказание
@app.post("/predict", response_model=PredictionOutput)
def predict(passenger: PassengerInput):
    X = prepare_features(passenger)

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