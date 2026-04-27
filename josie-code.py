import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from itertools import product


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


### 3. baseline SVM model

baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.8,
        min_df=2,
        ngram_range=(1,2)
    )),
    ("svm", LinearSVC(class_weight="balanced"))
])

# train baseline
baseline_model.fit(X_train, y_train)

# evaluate baseline on test set
y_pred_baseline = baseline_model.predict(X_test)

print("\n BASELINE RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print(classification_report(y_test, y_pred_baseline))


### 4. sample predictions

sample_reviews = [
    "The food was amazing and service was great",
    "Terrible experience, I will never come back",
    "It was okay, nothing special"
]

print("\nSample predictions:", baseline_model.predict(sample_reviews))


### 5. dev set hyperparameter tuning

param_grid = {
    "max_df": [0.7, 0.8, 0.9],
    "ngram_range": [(1,1), (1,2)],
    "C": [0.1, 1, 10]
}

best_score = 0
best_params = None
best_model = None

for max_df, ngram_range, C in product(
    param_grid["max_df"],
    param_grid["ngram_range"],
    param_grid["C"]
):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=max_df,
            min_df=2,
            ngram_range=ngram_range
        )),
        ("svm", LinearSVC(C=C, class_weight="balanced"))
    ])
    
    # train on train only
    model.fit(X_train, y_train)
    
    # evaluate on dev
    y_dev_pred = model.predict(X_dev)
    score = accuracy_score(y_dev, y_dev_pred)
    
    if score > best_score:
        best_score = score
        best_params = (max_df, ngram_range, C)
        best_model = model

print("\n DEV TUNING RESULTS")
print("Best Dev Accuracy:", best_score)
print(f"Best Params: max_df={best_params[0]}, ngram_range={best_params[1]}, C={best_params[2]}")


### 6. retrain on train and dev

X_final = pd.concat([X_train, X_dev])
y_final = pd.concat([y_train, y_dev])

best_model.fit(X_final, y_final)


### 7. final test evaluation

y_test_pred = best_model.predict(X_test)

print("\n FINAL MODEL RESULTS")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


### 8. confusion matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


### 9. feature interpretation

feature_names = best_model.named_steps["tfidf"].get_feature_names_out()
coefficients = best_model.named_steps["svm"].coef_[0]

top_positive = np.argsort(coefficients)[-10:]
top_negative = np.argsort(coefficients)[:10]

print("\nTop Positive Words:")
for i in reversed(top_positive):
    print(feature_names[i], coefficients[i])

print("\nTop Negative Words:")
for i in top_negative:
    print(feature_names[i], coefficients[i])
