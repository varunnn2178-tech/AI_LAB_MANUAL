import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 1. LOAD DATA
# =============================
CSV_PATH = "cleaned.csv"   # <-- change this to your CSV file name
df = pd.read_csv(CSV_PATH)

print("Columns in CSV:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# =============================
# 2. OPTIONAL: SUBSAMPLE (for speed)
# =============================
MAX_ROWS = 50000   # adjust if needed (e.g. 20000, 30000)
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"\nSubsampled dataset to {MAX_ROWS} rows for faster training.")
else:
    print(f"\nDataset has only {len(df)} rows, no subsampling applied.")

# =============================
# 3. TARGET & FEATURES
# =============================
TARGET_COL = "signature"

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in CSV!")

X = df.drop(columns=[TARGET_COL]).copy()
y = df[TARGET_COL].astype(str)

# =============================
# 4. ENCODE FEATURES & TARGET
# =============================
feature_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =============================
# 5. TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("\nShapes after split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# =============================
# 6. HELPER: EVALUATE + PLOT CONFUSION MATRIX
# =============================
def evaluate_model(name, model, X_test, y_test, top_n_classes=15):
    print("\n" + "=" * 60)
    print(f"=== {name} ===")
    print("=" * 60)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics (using encoded labels)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report (encoded labels):\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Full confusion matrix (might be huge)
    cm_full = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (encoded labels) - top-left 20x20 block:")
    print(cm_full[:20, :20])
    
    # ---------- Graphical Confusion Matrix (Heatmap) ----------
    # We can't plot all 600+ classes, so show only the top N most frequent classes
    value_counts = pd.Series(y_test).value_counts()
    top_classes = value_counts.index[:top_n_classes]

    cm_top = confusion_matrix(y_test, y_pred, labels=top_classes)

    # Decode class ids back to original signature strings for axis labels
    class_names = label_encoder.inverse_transform(top_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_top,
        annot=False,  # change to True if you want numbers in each cell
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{name} - Confusion Matrix (Top {top_n_classes} Classes)")
    plt.tight_layout()
    plt.show()
    
    # ---------- Example decoded predictions ----------
    try:
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        print("\nExample predictions (True vs Predicted signatures):")
        for t, p in list(zip(y_test_labels, y_pred_labels))[:10]:
            print(f"True: {t}  |  Pred: {p}")
    except Exception as e:
        print("\nCould not decode labels back to strings:", e)


# =============================
# 7. MODEL 1: GAUSSIAN NAIVE BAYES
# =============================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
evaluate_model("Gaussian Naive Bayes", nb_model, X_test, y_test)

# =============================
# 8. MODEL 2: DECISION TREE
# =============================
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=15,
    min_samples_leaf=5,
    random_state=42
)
dt_model.fit(X_train, y_train)
evaluate_model("Decision Tree", dt_model, X_test, y_test)

# =============================
# 9. MODEL 3: LOGISTIC REGRESSION
# =============================
log_model = LogisticRegression(
    solver="lbfgs",
    max_iter=200,
    n_jobs=-1,
    verbose=0
)
print("\nTraining Logistic Regression (may take a bit)...")
log_model.fit(X_train, y_train)
evaluate_model("Logistic Regression", log_model, X_test, y_test)

print("\n=== DONE (Classification Matrices + Graphical Confusion Matrices for 3 Models) ===")
