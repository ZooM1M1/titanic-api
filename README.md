# Titanic Survival API

FastAPI-сервис для предсказания выживания на Титанике.

## Установка

pip install -r requirements.txt

## Обучение модели

python train_model.py

## Запуск сервера

uvicorn main:app --reload

Открыть: http://127.0.0.1:8000/docs

## Пример запроса

POST /predict

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 30,
  "fare": 100,
  "sibsp": 0,
  "parch": 0,
  "embarked": "C",
  "title": "Mrs"
}

Ответ:

{
  "survived": 1,
  "probability": 0.978,
  "message": "Passenger would survive with probability 97.8%"
}

Технологии

Python, scikit-learn, FastAPI, Pydantic, joblib, uvicorn.

