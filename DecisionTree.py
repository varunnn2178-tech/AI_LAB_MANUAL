import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------
CSV_PATH = "cleaned.csv"   # <-- change this if needed
df = pd.read_csv(CSV_PATH)

print("Columns in CSV:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------
# 2. Subsample for speed (optional)
# ---------------------------------------------------
MAX_ROWS = 50000   # try 20000 / 50000 etc. depending on speed
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    print(f"\nSubsampled dataset to {MAX_ROWS} rows for faster training.")
else:
    print(f"\nDataset has only {len(df)} rows, no subsampling applied.")

# ---------------------------------------------------
# 3. Target column
# ---------------------------------------------------
target = "signature"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in CSV!")

y_raw = df[target].astype(str)
X = df.drop(columns=[target]).copy()

# ---------------------------------------------------
# 4. Encode target labels
# ---------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("\nNumber of classes in target:", len(label_encoder.classes_))

# ---------------------------------------------------
# 5. Choose numeric & categorical features
# ---------------------------------------------------
candidate_numeric = [
    "clamav",
    "vtpercent",
    "vtpercent_ratio",
    "first_seen_year",
    "first_seen_month",
    "first_seen_day",
    "has_signature",
    "has_clamav",
]
numeric_features = [c for c in candidate_numeric if c in X.columns]

candidate_categorical = [
    "reporter",
    "file_type_guess",
    "mime_type",
    "file_ext",
    "label",          # optional – if present
]
categorical_features = [c for c in candidate_categorical if c in X.columns]

print("\nNumeric feature columns:", numeric_features)
print("Categorical feature columns:", categorical_features)

if not numeric_features and not categorical_features:
    raise SystemExit("No usable features found – check your column names.")

X = X[numeric_features + categorical_features].copy()

# ---------------------------------------------------
# 6. Train/test split
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

print("\nShapes after split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ---------------------------------------------------
# 7. Preprocessing pipelines
# ---------------------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),  # sparse by default
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# ---------------------------------------------------
# 8. Full Decision Tree pipeline
# ---------------------------------------------------
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("dt", dt),
    ]
)

print("\nTraining Decision Tree pipeline...")
model.fit(X_train, y_train)

# ---------------------------------------------------
# 9. Predict & evaluate
# ---------------------------------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nDecision Tree Accuracy:", acc)

print("\nClassification Report (encoded labels):\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------------------------
# 10. Confusion Matrix – TOP N CLASSES (readable plot)
# ---------------------------------------------------
TOP_N = 20  # change this to 10, 30, etc. as you like

test_counts = Counter(y_test)
top_classes_int = [cls for cls, _ in test_counts.most_common(TOP_N)]

cm_top = confusion_matrix(y_test, y_pred, labels=top_classes_int)
top_class_names = label_encoder.inverse_transform(top_classes_int)

print(f"\nPlotting confusion matrix for top {TOP_N} classes by frequency.")
print("Top class names:", list(top_class_names))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_top,
    display_labels=top_class_names,
)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
plt.xticks(rotation=45, ha="right")
plt.title(f"Decision Tree Confusion Matrix (Top {TOP_N} signatures)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 11. Example predictions (decoded)
# ---------------------------------------------------
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nExample predictions (True vs Predicted signatures):")
for true_label, pred_label in list(zip(y_test_labels, y_pred_labels))[:10]:
    print(f"True: {true_label}  |  Pred: {pred_label}")

# ---------------------------------------------------
# 12. Feature Importances (using preprocessor feature names)
# ---------------------------------------------------
fitted_preproc = model.named_steps["preprocessor"]
fitted_dt = model.named_steps["dt"]

# THIS is the important fix: get exact output feature names
feature_names = fitted_preproc.get_feature_names_out()
importances = fitted_dt.feature_importances_

print("\nLength of feature_names:", len(feature_names))
print("Length of importances  :", len(importances))

feat_imp_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTop 20 feature importances:")
print(feat_imp_df.head(20))

plt.figure(figsize=(12, 6))
sns.barplot(
    x="importance",
    y="feature",
    data=feat_imp_df.head(20),
)
plt.title("Top 20 Decision Tree Feature Importances (signature)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 13. Tree Visualization (optional – can be big)
# ---------------------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    fitted_dt,
    feature_names=feature_names,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=6,
)
plt.title("Decision Tree Visualization (signature)")
plt.tight_layout()
plt.show()

print("\n=== DONE (Decision Tree with Top-N confusion matrix & fixed feature names) ===")
