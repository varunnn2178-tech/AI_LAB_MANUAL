import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# =========================
# 1. LOAD DATA
# =========================
CSV_PATH = "cleaned.csv"  # <-- change this to your CSV file name

df = pd.read_csv(CSV_PATH)
print("Columns in CSV:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# =========================
# 2. COMMON SETTINGS
# =========================
TARGET_COL = "signature"    # label column for parts 3 and 4
TEXT_COL = "file_name"      # text feature for TF-IDF model (change if you want)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in CSV!")

# ==========================================================
# 3. GAUSSIAN NAIVE BAYES ON ENCODED TABULAR FEATURES
#    (existing functionality)
# ==========================================================

print("\n=== GAUSSIAN NAIVE BAYES ON ENCODED FEATURES ===")

# Split features (X) and target (y)
X_tab = df.drop(columns=[TARGET_COL]).copy()
y_raw = df[TARGET_COL].astype(str)

# Encode all feature columns (most of your columns are text/hashes)
feature_encoders = {}
for col in X_tab.columns:
    le = LabelEncoder()
    X_tab[col] = le.fit_transform(X_tab[col].astype(str))
    feature_encoders[col] = le

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# Train-test split (no stratify because some classes may have only 1 sample)
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X_tab, y_encoded, test_size=0.2, random_state=42
)

# Train GaussianNB
gauss_model = GaussianNB()
gauss_model.fit(X_train_tab, y_train_tab)

# Predict
y_pred_tab = gauss_model.predict(X_test_tab)

# Evaluation
acc_gauss = accuracy_score(y_test_tab, y_pred_tab)
print(f"\nGaussianNB Accuracy: {acc_gauss:.4f}")
print("\nGaussianNB Classification Report (encoded labels):")
print(classification_report(y_test_tab, y_pred_tab, zero_division=0))

cm_gauss = confusion_matrix(y_test_tab, y_pred_tab)
print("\nGaussianNB Confusion Matrix:")
print(cm_gauss)

# ==========================================================
# 4. MULTINOMIAL NB + TF-IDF ON A TEXT COLUMN
#    (existing functionality)
# ==========================================================

print("\n=== MULTINOMIAL NAIVE BAYES WITH TF-IDF (REDUCED DATA) ===")

from collections import Counter

if TEXT_COL not in df.columns:
    raise ValueError(f"Text column '{TEXT_COL}' not found in CSV!")

# We reuse y_encoded from above (encoded 'signature' labels)
y_all = y_encoded
X_all_text = df[TEXT_COL].astype(str)

# 1) Keep only top N most frequent classes
TOP_N_CLASSES = 20   # you can change this (e.g., 10, 30, etc.)

class_counts = Counter(y_all)
top_classes = [cls for cls, _ in class_counts.most_common(TOP_N_CLASSES)]

mask = np.isin(y_all, top_classes)
X_text = X_all_text[mask]
y_text = y_all[mask]

print(f"Total samples after filtering to top {TOP_N_CLASSES} classes: {len(X_text)}")

# 2) Optional: limit max number of samples to avoid huge memory use
MAX_SAMPLES = 20000   # adjust as needed
if len(X_text) > MAX_SAMPLES:
    X_text, _, y_text, _ = train_test_split(
        X_text, y_text,
        train_size=MAX_SAMPLES,
        random_state=42,
        stratify=y_text
    )
    print(f"Subsampled to {MAX_SAMPLES} samples for TF-IDF model.")

# 3) Train-test split
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
)

# 4) TF-IDF + MultinomialNB pipeline (smaller feature space)
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=2000)),  # smaller vocabulary
    ("clf", MultinomialNB())
])

# 5) Train text model
text_pipeline.fit(X_train_text, y_train_text)

# 6) Predict
y_pred_text = text_pipeline.predict(X_test_text)

# 7) Evaluation
acc_text = accuracy_score(y_test_text, y_pred_text)
print(f"\nMultinomialNB (TF-IDF on '{TEXT_COL}') Accuracy: {acc_text:.4f}")

print("\nMultinomialNB Classification Report (encoded labels, reduced subset):")
print(classification_report(y_test_text, y_pred_text, zero_division=0))

cm_text = confusion_matrix(y_test_text, y_pred_text)
print("\nMultinomialNB Confusion Matrix (reduced subset):")
print(cm_text)

# ==========================================================
# 5. NEW: NAIVE BAYES CLASSIFICATION ON STRUCTURED FEATURES
#    WITH PREPROCESSING + GRAPHICAL OUTPUT (FROM FIRST CODE)
# ==========================================================

print("\n=== STRUCTURED GAUSSIAN NAIVE BAYES WITH PREPROCESSING (LABEL TARGET) ===")

if "label" in df.columns:
    # Map label to numeric
    label_mapping = {"benign": 0, "suspicious": 1, "malicious": 2}
    df_nb = df[df["label"].notna()].copy()
    df_nb["label_num"] = df_nb["label"].map(label_mapping)

    # Drop rows where mapping failed
    df_nb = df_nb[df_nb["label_num"].notna()].copy()
    y_nb = df_nb["label_num"].astype(int)

    # Numeric and categorical features (like your first script, but only if they exist)
    base_numeric_features = [
        "entropy",
        "pe_sections_count",
        "av_count",
        "duration_days",
        "log_file_size_winsor",
    ]
    # include tag_ indicator columns if present
    tag_features = [c for c in df_nb.columns if c.startswith("tag_")]
    numeric_features = [c for c in base_numeric_features if c in df_nb.columns] + tag_features

    # Categorical features
    base_cat_features = ["file_type_guess", "file_type", "family", "packed"]
    categorical_features = [c for c in base_cat_features if c in df_nb.columns]

    if not numeric_features and not categorical_features:
        print("No suitable numeric/categorical features found for structured NB. Skipping this part.")
    else:
        X_nb = df_nb[numeric_features + categorical_features].copy()

        print("Numeric features (NB structured):", numeric_features)
        print("Categorical features (NB structured):", categorical_features)

        # Train/test split
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
            X_nb,
            y_nb,
            test_size=0.2,
            random_state=42,
            stratify=y_nb,
        )
        print("Train/Test shapes (structured NB):", X_train_nb.shape, X_test_nb.shape)

        # Preprocessing pipeline (same logic as first script)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor_nb = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Preprocess train/test
        X_train_nb_proc = preprocessor_nb.fit_transform(X_train_nb)
        X_test_nb_proc = preprocessor_nb.transform(X_test_nb)
        print("Processed feature shape (structured NB):", X_train_nb_proc.shape)

        # Gaussian Naive Bayes
        nb_struct = GaussianNB()
        nb_struct.fit(X_train_nb_proc, y_train_nb)

        y_pred_nb = nb_struct.predict(X_test_nb_proc)
        y_prob_nb = nb_struct.predict_proba(X_test_nb_proc)

        # Confusion Matrix with display
        cm_nb = confusion_matrix(y_test_nb, y_pred_nb, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_nb,
            display_labels=list(label_mapping.keys())
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Naive Bayes Confusion Matrix (Structured Features)")
        plt.tight_layout()
        plt.show()

        # Classification report
        print("\nStructured Naive Bayes Classification Report:\n")
        print(
            classification_report(
                y_test_nb,
                y_pred_nb,
                target_names=list(label_mapping.keys()),
                zero_division=0,
            )
        )

        # Plot class probability distributions (like first code)
        plt.figure(figsize=(8, 5))
        for i, class_name in enumerate(label_mapping.keys()):
            plt.hist(
                y_prob_nb[:, i],
                bins=20,
                alpha=0.5,
                label=class_name,
            )
        plt.xlabel("Predicted class probability")
        plt.ylabel("Count")
        plt.title("Naive Bayes predicted probabilities per class (Structured)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Approximate feature importance (means per class)
        means = nb_struct.theta_ if hasattr(nb_struct, "theta_") else None
        if means is not None:
            plt.figure(figsize=(12, 6))
            for i, class_name in enumerate(label_mapping.keys()):
                plt.plot(means[i], label=class_name)
            plt.title("Approximate GaussianNB Feature Means per Class (Structured)")
            plt.xlabel("Feature index after preprocessing")
            plt.ylabel("Mean value")
            plt.legend()
            plt.tight_layout()
            plt.show()

        print("Structured Naive Bayes classification complete.")
else:
    print("\nColumn 'label' not found in CSV. Skipping structured Naive Bayes part.")
