# 🚢 Titanic Survival Prediction API (v2.0)

Высокопроизводительный REST API на базе **FastAPI** и промышленного **Scikit-Learn Pipeline** для прогнозирования вероятности выживания пассажиров Титаника.

---

## 🏗️ Архитектура сервиса (v2.0 Monolithic Pipeline)

В версии **2.0** микросервис полностью переведен на архитектуру единого конвейера (`ColumnTransformer` + `Pipeline`). Больше нет ручной предобработки, жестко закодированных бинарных колонок и риска несовпадения признаков (Data Drift / Mismatch).

```text
HTTP Request (JSON)
       │
       ▼
PassengerInput (Pydantic v2 Schema)
       │
       ▼
prepare_features() (DataFrame из сырых значений)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Scikit-Learn Monolithic Pipeline                    │
│                                                                        │
│   ┌─── Числовые признаки ───> SimpleImputer(median) ─> StandardScaler  │
│   │   (Pclass, Age, Fare, FamilySize, IsAlone)                         │
│ ──┤                                                                    │
│   └─── Категории ───────────> SimpleImputer(most_frequent) ─> OneHot   │
│       (Sex, Embarked, Title)                                           │
│                                                                        │
│                                 │                                      │
│                                 ▼                                      │
│                  LogisticRegression (max_iter=1000)                    │
└────────────────────────────────────────────────────────────────────────┘
       │
       ▼
PredictionOutput { survived, probability, message }
```

### Преимущества архитектуры:
- **Защита от сбоев**: `OneHotEncoder(handle_unknown="ignore")` предотвращает ошибки сервера при получении неизвестных категорий или редких титулов.
- **Автономный импьютинг**: `SimpleImputer` автоматически заполняет пропущенные значения (`Age`, `Embarked`) на основе статистик обучающей выборки.
- **Единый артефакт**: Модель и весь препроцессинг упакованы в один файл `model_pipeline.joblib`.

---

## 🛠️ Стек технологий
- **Фреймворк**: [FastAPI](https://fastapi.tiangolo.com/) (асинхронный высокопроизводительный веб-сервер)
- **Валидация данных**: [Pydantic v2](https://docs.pydantic.dev/) (строгая типизация входных и выходных DTO)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `LogisticRegression`)
- **Сериализация**: [Joblib](https://joblib.readthedocs.io/)
- **ASGI-сервер**: [Uvicorn](https://www.uvicorn.org/)

---

## 🚀 Быстрый старт

### 1. Клонирование и установка зависимостей
```bash
git clone https://github.com/ZooM1M1/titanic-api.git
cd titanic-api
pip install -r requirements.txt
```

### 2. Обучение пайплайна
Скрипт загружает данные, производит Feature Engineering (`Title`, `FamilySize`, `IsAlone`), обучает монолитный Pipeline и сериализует его в `model_pipeline.joblib`:
```bash
python train_model.py
```

### 3. Запуск веб-сервера
```bash
uvicorn main:app --reload
```
Сервер будет доступен по адресу: `http://127.0.0.1:8000`

---

## 📖 Документация API и интерактивный Swagger

FastAPI автоматически генерирует интерактивную OpenAPI-документацию:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 Эндпоинты

### `GET /`
Проверка работоспособности сервиса (Health Check).

### `POST /predict`
Основной эндпоинт для вычисления вероятности выживания.

#### Пример тела запроса (Request Body):
```json
{
  "pclass": 1,
  "sex": "female",
  "age": 30.0,
  "fare": 100.0,
  "sibsp": 0,
  "parch": 0,
  "embarked": "C",
  "title": "Mrs"
}
```

#### Пример ответа (Response Body):
```json
{
  "survived": 1,
  "probability": 0.976,
  "message": "Passenger would survive with probability 97.6%"
}
```

---

## 🧪 Тестирование
Для проверки эндпоинта предусмотрен тестовый скрипт с несколькими профилями пассажиров:
```bash
python test_api.py
```
