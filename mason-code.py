import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

### 3. baseline RNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

# convert labels to binary
y_train_bin = (y_train == "positive").astype(int)
y_dev_bin   = (y_dev == "positive").astype(int)
y_test_bin  = (y_test == "positive").astype(int)

# tokenize text
max_vocab = 5000
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_dev_seq   = tokenizer.texts_to_sequences(X_dev)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

# pad sequences
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_dev_pad   = pad_sequences(X_dev_seq, maxlen=max_len, padding='post')
X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# build model
baseline_rnn = Sequential([
    Embedding(input_dim=max_vocab, output_dim=64, input_length=max_len),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

baseline_rnn.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train
baseline_rnn.fit(
    X_train_pad, y_train_bin,
    epochs=5,
    batch_size=32,
    validation_data=(X_dev_pad, y_dev_bin)
)

# evaluate
loss, acc = baseline_rnn.evaluate(X_test_pad, y_test_bin)

print("\n BASELINE RNN RESULTS")
print("Test Accuracy:", acc)

### 4. sample predictions
def predict_sentiment(model, texts):
    seq = tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    preds = model.predict(pad)
    return ["positive" if p > 0.5 else "negative" for p in preds]

sample_reviews = [
    "The food was amazing and service was great",
    "Terrible experience, I will never come back",
    "It was okay, nothing special"
]

print("\nSample predictions:", predict_sentiment(baseline_rnn, sample_reviews))


### 5. dev set hyperparameter tuning
embedding_dims = [32, 64]
rnn_units = [32, 64]
batch_sizes = [32, 64]

best_score = 0
best_config = None
best_model = None

for emb_dim, rnn_unit, batch_size in product(embedding_dims, rnn_units, batch_sizes):
    
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=emb_dim, input_length=max_len),
        SimpleRNN(rnn_unit),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.fit(
        X_train_pad, y_train_bin,
        epochs=3,  # shorter for tuning
        batch_size=batch_size,
        verbose=0
    )
    
    _, dev_acc = model.evaluate(X_dev_pad, y_dev_bin, verbose=0)
    
    if dev_acc > best_score:
        best_score = dev_acc
        best_config = (emb_dim, rnn_unit, batch_size)
        best_model = model

print("\n DEV TUNING RESULTS")
print("Best Dev Accuracy:", best_score)
print(f"Best Params: embedding_dim={best_config[0]}, rnn_units={best_config[1]}, batch_size={best_config[2]}")


### 6. retrain on train and dev
X_final = np.concatenate([X_train_pad, X_dev_pad])
y_final = np.concatenate([y_train_bin, y_dev_bin])

final_model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=best_config[0], input_length=max_len),
    SimpleRNN(best_config[1]),
    Dense(1, activation='sigmoid')
])

final_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

final_model.fit(
    X_final, y_final,
    epochs=5,
    batch_size=best_config[2])


### 7. final test evaluation
loss, acc = final_model.evaluate(X_test_pad, y_test_bin)

print("\n FINAL MODEL RESULTS")
print("Test Accuracy:", acc)
### 8. confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred_probs = final_model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred))