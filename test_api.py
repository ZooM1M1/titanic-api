"""
Тестирование API через Python.
Запуск: python test_api.py  (сервер uvicorn должен быть запущен).
"""

import requests


URL = "http://127.0.0.1:8000/predict"

passengers = [
    {
        "pclass": 3, "sex": "male", "age": 22, "fare": 7.25,
        "sibsp": 0, "parch": 0, "embarked": "S", "title": "Mr",
    },
    {
        "pclass": 1, "sex": "female", "age": 30, "fare": 100,
        "sibsp": 0, "parch": 0, "embarked": "C", "title": "Mrs",
    },
    {
        "pclass": 3, "sex": "female", "age": 5, "fare": 15,
        "sibsp": 1, "parch": 2, "embarked": "S", "title": "Miss",
    },
]

for p in passengers:
    response = requests.post(URL, json=p)
    result = response.json()
    print(f"\nВход: pclass={p['pclass']}, sex={p['sex']}, age={p['age']}, title={p['title']}")
    print(f"Ответ: {result['message']}")
    print(f"  survived={result['survived']}, probability={result['probability']:.3f}")