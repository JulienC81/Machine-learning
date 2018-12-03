# coding: utf-8

import pandas as pd
from sklearn import preprocessing
from sklearn import cross_validation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR

df_data = pd.read_csv(
    r"Stations-Total\train.csv", sep=";", decimal=",", encoding="utf-8"
)

df_test = pd.read_csv(
    r"Stations-Total\test.csv", sep=";", decimal=",", encoding="utf-8"
)

columns_to_drop = ["id", "article_nom"] + df_data.columns[-12:].tolist()
df_data = df_data.drop(columns_to_drop, axis=1)

to_category = [
    "implant",
    "id_categorie_6",
    "id_categorie_5",
    "id_categorie_4",
    "cat6_nom",
    "cat5_nom",
    "cat4_nom",
]

df_data["date"] = pd.to_datetime(df_data["date"])
df_data["year"] = [d.year for d in df_data["date"]]
df_data["month"] = [d.month for d in df_data["date"]]
df_data["day"] = [d.day for d in df_data["date"]]

le = preprocessing.LabelEncoder()

for col in to_category:
    le.fit(df_data[col])
    df_data[col + "_cat"] = le.transform(df_data[col])

df_data = df_data.drop(to_category, axis=1)




print(df_data.tail(20))
