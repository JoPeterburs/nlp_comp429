import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


### 1. load data

df = pd.read_csv('data/Restaurant reviews.csv')
print(df.head())
print(df.columns)

# select review and rating columns
reviews = df[['Review', 'Rating']].copy()
# drop na
reviews = reviews.dropna(subset=["Review", "Rating"])

# change rating column to numeric
reviews['Rating'] = pd.to_numeric(reviews['Rating'], errors='coerce')

# add positive (1)/negative (0) labels
reviews["Sentiment"] = reviews["Rating"].apply(
    lambda x: "positive" if x >= 4 else "negative" 
)

print(reviews.head())
print(reviews["Sentiment"].value_counts())


### 2. train/dev/test sets (70/15/15)

X = reviews["Review"]
y = reviews["Sentiment"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,   # 30% goes to temp (dev + test)
    random_state=19,
    stratify=y
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,   # split temp evenly → 15% dev, 15% test
    random_state=19,
    stratify=y_temp
)

# check sizes
print("Train:", len(X_train))
print("Dev:", len(X_dev))
print("Test:", len(X_test))
