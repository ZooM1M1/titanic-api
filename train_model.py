import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df["Title"] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
common_titles = ["Mr", 'Miss', 'Mrs', 'Master']
df['Title'] = df['Title'].where(df['Title'].isin(common_titles), 'Rare')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']]

df=pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=False)

X=df.drop(columns=['Survived'])
y=df['Survived']

pipe= Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])

pipe.fit(X, y)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
joblib.dump(pipe, os.path.join(BASE_DIR, "model.joblib"))
joblib.dump(list(X.columns), os.path.join(BASE_DIR, "columns.joblib"))

print('Модель сохранена: model.joblib')
print('Колонки сохранены: columns.joblib')
print('Признаков:', X.shape[1])