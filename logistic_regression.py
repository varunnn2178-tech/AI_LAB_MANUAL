import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ---------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------
CSV_PATH = "cleaned.csv"   # change if needed
df = pd.read_csv(CSV_PATH)

print("Columns in CSV:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------
# 2. (Optional) Subsample to speed up training
# ---------------------------------------------------
MAX_ROWS = 50000  # adjust based on your machine
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"\nSubsampled dataset to {MAX_ROWS} rows for faster training.")
else:
    print(f"\nDataset has only {len(df)} rows, no subsampling applied.")

# ---------------------------------------------------
# 3. Target column
# ---------------------------------------------------
TARGET = "signature"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in CSV!")

y_raw = df[TARGET].astype(str)

# encode target labels -> integers 0..(n_classes-1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("\nNumber of classes in target:", len(label_encoder.classes_))

# ---------------------------------------------------
# 4. Select sensible features (avoid huge hashes, file_name, etc.)
# ---------------------------------------------------
candidate_numeric = [
    "vtpercent",
    "vtpercent_ratio",
    "first_seen_year",
    "first_seen_month",
    "first_seen_day",
    "has_signature",
    "has_clamav",
    # "clamav",  # often all NaN, skip
]
numeric_features = [c for c in candidate_numeric if c in df.columns]

candidate_categorical = [
    "reporter",
    "file_type_guess",
    "mime_type",
    "file_ext",
    "label",  # optional malware label (benign/suspicious/malicious/unknown)
]
categorical_features = [c for c in candidate_categorical if c in df.columns]

print("\nNumeric feature columns:", numeric_features)
print("Categorical feature columns:", categorical_features)

if not numeric_features and not categorical_features:
    raise SystemExit("No usable feature columns found – check your CSV.")

X = df[numeric_features + categorical_features].copy()

# ---------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------
value_counts = pd.Series(y).value_counts()
min_count = value_counts.min()
if min_count < 2:
    print("Not using stratify because some classes have only 1 sample.")
    stratify_arg = None
else:
    stratify_arg = y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_arg,
)

print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

# ---------------------------------------------------
# 6. Preprocessing pipelines
# ---------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),  # sparse by default
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# ---------------------------------------------------
# 7. Full Logistic Regression pipeline
# ---------------------------------------------------
log_reg = LogisticRegression(
    max_iter=500,
    multi_class="auto",      # avoid future deprecation warning
    solver="lbfgs",
    n_jobs=-1,
    verbose=0,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("logreg", log_reg),
    ]
)

print("\nTraining Logistic Regression (this may take a bit)...")
model.fit(X_train, y_train)

# ---------------------------------------------------
# 8. Predictions & probabilities
# ---------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # shape: (n_samples, n_trained_classes)

# Important: classes used by the trained logistic regression
logreg_fitted = model.named_steps["logreg"]
logreg_classes = logreg_fitted.classes_   # e.g. array([  2,  4,  9, ...])
print("\nNumber of classes learned by LogisticRegression:", len(logreg_classes))

# ---------------------------------------------------
# 9. Basic evaluation
# ---------------------------------------------------
acc = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", acc)

print("\nClassification Report (per encoded class):\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------
# 10. Confusion Matrix for TOP-N classes (readable)
# ---------------------------------------------------
TOP_N_CM = 20  # number of most frequent signatures to show

test_counts = Counter(y_test)
top_classes_int = [cls for cls, _ in test_counts.most_common(TOP_N_CM)]
top_class_names = label_encoder.inverse_transform(top_classes_int)

cm_top = confusion_matrix(y_test, y_pred, labels=top_classes_int)

print(f"\nPlotting confusion matrix for top {TOP_N_CM} classes by frequency.")
print("Top class names:", list(top_class_names))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_top,
    display_labels=top_class_names,
)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
plt.xticks(rotation=45, ha="right")
plt.title(f"Logistic Regression Confusion Matrix (Top {TOP_N_CM} signatures)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 11. ROC Curves for a few top classes (using correct column indices)
# ---------------------------------------------------
TOP_N_ROC = 3  # only plot ROC for a few key classes

# Only keep top classes that actually exist in the trained model
valid_top_classes = [cls for cls in top_classes_int if cls in logreg_classes]

if not valid_top_classes:
    print("\nNo top classes present in trained model for ROC curves; skipping ROC section.")
else:
    colors = ["blue", "green", "red", "orange", "purple"]

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(valid_top_classes[:TOP_N_ROC]):
        # find which column in y_prob corresponds to this class
        col_idx = np.where(logreg_classes == cls)[0][0]

        y_true_bin = (y_test == cls).astype(int)
        y_score = y_prob[:, col_idx]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        name = label_encoder.inverse_transform([cls])[0]

        plt.plot(
            fpr,
            tpr,
            lw=2,
            color=colors[i % len(colors)],
            label=f"{name} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curves (Top classes)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# 12. Probability distribution plots for the same classes
# ---------------------------------------------------
TOP_N_HIST = 3

if not valid_top_classes:
    print("\nNo valid top classes for probability histograms; skipping hist section.")
else:
    plt.figure(figsize=(8, 5))
    for i, cls in enumerate(valid_top_classes[:TOP_N_HIST]):
        col_idx = np.where(logreg_classes == cls)[0][0]
        name = label_encoder.inverse_transform([cls])[0]

        plt.hist(
            y_prob[:, col_idx],
            bins=20,
            alpha=0.5,
            label=name,
        )

    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Logistic Regression Predicted Probabilities (Top classes)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# 13. Example decoded predictions
# ---------------------------------------------------
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nExample predictions (True vs Predicted signatures):")
for true_label, pred_label in list(zip(y_test_labels, y_pred_labels))[:10]:
    print(f"True: {true_label}  |  Pred: {pred_label}")

print("\n=== DONE (Logistic Regression with preprocessing + graphics, fixed ROC indexing) ===")
