import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   # NEW: for heatmap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =============================
# 1. LOAD DATA
# =============================
CSV_PATH = "cleaned.csv"   # <-- change this to your CSV file name
df = pd.read_csv(CSV_PATH)

print("Columns in CSV:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# =============================
# 2. TARGET VARIABLE
# =============================
TARGET_COL = "signature"

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found!")

# Drop rows with missing target
df = df[df[TARGET_COL].notna()]
print(f"\nRows remaining after dropping NaN in signature: {len(df)}")

# =============================
# 3. SUBSAMPLE FOR SPEED
# =============================
MAX_ROWS = 50000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"Subsampled dataset to {MAX_ROWS} rows.")
else:
    print("Dataset small enough — no sampling.")

# =============================
# 4. SPLIT FEATURES + TARGET
# =============================
y_raw = df[TARGET_COL].astype(str)
X = df.drop(columns=[TARGET_COL]).copy()

# Remove columns that are entirely NaN
X = X.dropna(axis=1, how="all")

print("\nFeature columns:")
print(X.columns)

# =============================
# 5. ENCODE CATEGORICAL FEATURES
# =============================
for col in X.columns:
    if X[col].dtype == "object":
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))

# =============================
# 6. IMPUTE MISSING VALUES
# =============================
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# =============================
# 7. SCALE FEATURES
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("\nAny NaN in X_scaled?", np.isnan(X_scaled).any())

# =============================
# 8. ENCODE TARGET
# =============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

num_classes = len(np.unique(y_encoded))
print("Number of classes:", num_classes)

# =============================
# 9. TRAIN/TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# =============================
# 10. BUILD ANN (3-LAYER FEED-FORWARD)
# =============================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel summary:")
model.summary()

# =============================
# 11. TRAIN NETWORK (BACKPROP)
# =============================
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("\nTraining ANN...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.1,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# =============================
# 12. PLOT TRAINING GRAPHS
# =============================

# Accuracy
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("ANN Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Loss
plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("ANN Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# 13. EVALUATE ON TEST SET
# =============================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n=== ANN RESULTS ===")
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# =============================
# 14. PREDICTIONS & METRICS
# =============================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report (encoded labels):")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix Shape:", cm.shape)

# =============================
# 15. GRAPHICAL CONFUSION MATRIX (TOP N CLASSES)
# =============================
# We have many signature classes, so plot only top N most frequent
TOP_N = 15

# Get top-N frequent classes in the test set
value_counts = pd.Series(y_test).value_counts()
top_classes = value_counts.index[:TOP_N]

cm_top = confusion_matrix(y_test, y_pred, labels=top_classes)

# Decode encoded class IDs back to original signature names
class_names = label_encoder.inverse_transform(top_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_top,
    annot=False,      # set True if you want counts inside cells
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.xlabel("Predicted Signature")
plt.ylabel("True Signature")
plt.title(f"ANN Confusion Matrix (Top {TOP_N} Signature Classes)")
plt.tight_layout()
plt.show()

# =============================
# 16. EXAMPLE DECODED PREDICTIONS
# =============================
print("\nExample predictions (True vs Predicted signatures):")
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

for t, p in list(zip(y_test_labels, y_pred_labels))[:10]:
    print(f"True: {t}  |  Pred: {p}")

print("\n=== DONE (Feedforward ANN with Backprop + Confusion Matrix Plot) ===")
