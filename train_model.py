import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1. Загрузка данных
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Создаем производные признаки (Feature Engineering)
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
common_titles = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 3. Отбираем сырые колонки (обрати внимание: НИКАКОГО fillna и get_dummies вручную!)
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "FamilySize", "IsAlone"]
X = df[features]
y = df["Survived"]

# 4. Разделяем колонки по типам
numeric_features = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone"]
categorical_features = ["Sex", "Embarked", "Title"]

# 5. Собираем ветки препроцессинга
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# 6. Полный монолитный пайплайн: Препроцессинг + Модель
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# 7. Обучаем ВСЁ разом
full_pipeline.fit(X, y)

# 8. Сохраняем ТОЛЬКО ОДИН файл модели
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_pipeline.joblib")
joblib.dump(full_pipeline, model_path)

print(f"✅ Монолитный Pipeline успешно сохранен в: {model_path}")
print("Файл columns.joblib больше не требуется!")