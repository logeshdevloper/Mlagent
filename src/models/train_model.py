# train_model.py (LightGBM version)
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
from datetime import datetime
import os

# Load data
df = pd.read_csv("data/secondSet.csv")
X = df.drop(columns=["label","symbol","created_at"])
y = df["label"]

# Add this after: df = pd.read_csv("data/firstdataset.csv")
print("Label balance:")
print(df['label'].value_counts())
print("Percentages:")
print(df['label'].value_counts(normalize=True))


# Split chronologically (no shuffling)
n = len(df)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

# LightGBM model
model = lgb.LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',        # default, but explicit
    n_estimators=1_000,          # more trees
    learning_rate=0.02,          # smaller step
    num_leaves=63,               # ~2×(max_depth)
    max_depth=-1,                # let num_leaves control complexity
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=20,
    lambda_l1=0.0,
    lambda_l2=0.1,
    is_unbalance=True, 
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


# Train
model.fit(X_train, y_train, 
         eval_set=[(X_val, y_val)],
         callbacks=[lgb.early_stopping(40)])

# Evaluate
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.3f}")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.3f}")

# Save model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model_v1_lgb.pkl")
print("✅ Model saved!")
