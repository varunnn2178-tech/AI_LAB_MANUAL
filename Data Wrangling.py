"""
Extended MalwareBazaar CSV wrangling + ML-prep pipeline.

- Robust load (comments, malformed rows, etc.)
- Type cleaning & deduplication
- Missingness overview
- Feature engineering (vtpercent-based label, time features, file extension, flags)
- ML-ready preprocessing (imputation, scaling, encoding)
- Train/test split
- Optional visualizations

Requires:
    pip install --upgrade pandas scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
INPUT_FILE = "full.csv"       # original MalwareBazaar dump
OUTPUT_FILE = "cleaned.csv"   # cleaned + feature-engineered output
RANDOM_STATE = 42
PRINT_PREVIEW = True

pd.set_option("display.max_columns", 200)

# -------------------------------------------------------
# 1. DEFINE COLUMN NAMES (14 FIELDS)
# -------------------------------------------------------
col_names = [
    "first_seen_utc",
    "sha256_hash",
    "md5_hash",
    "sha1_hash",
    "reporter",
    "file_name",
    "file_type_guess",
    "mime_type",
    "signature",
    "clamav",
    "vtpercent",
    "imphash",
    "ssdeep",
    "tlsh",
]

# -------------------------------------------------------
# 2. LOAD CSV ROBUSTLY
# -------------------------------------------------------
df = pd.read_csv(
    INPUT_FILE,
    comment="#",
    names=col_names,
    header=None,
    engine="python",
    sep=",",
    quotechar='"',
    skipinitialspace=True,
    na_values=["n/a", "N/A", "NA", ""],
    on_bad_lines="skip",   # for pandas >= 1.4.0
)

if PRINT_PREVIEW:
    print("Loaded shape (rows, cols):", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nRaw head:")
    print(df.head())

# -------------------------------------------------------
# 3. STRIP WHITESPACE FROM STRING COLUMNS
# -------------------------------------------------------
obj_cols = df.select_dtypes(include="object").columns
for col in obj_cols:
    df[col] = df[col].str.strip()

# -------------------------------------------------------
# 4. CONVERT TYPES
# -------------------------------------------------------

# first_seen_utc → datetime
df["first_seen_utc"] = pd.to_datetime(df["first_seen_utc"], errors="coerce")

# vtpercent → float
df["vtpercent"] = pd.to_numeric(df["vtpercent"], errors="coerce")

# -------------------------------------------------------
# 5. MISSINGNESS OVERVIEW
# -------------------------------------------------------
missing = df.isna().sum().sort_values(ascending=False)
print("\nMissing values per column:")
print(missing[missing > 0])

# -------------------------------------------------------
# 6. DROP BAD / DUPLICATE ROWS
# -------------------------------------------------------

# Drop rows without sha256_hash (critical identifier)
before = len(df)
df = df.dropna(subset=["sha256_hash"])
if PRINT_PREVIEW:
    print(f"\nDropped rows with missing sha256_hash: {before - len(df)}")

# Drop duplicate sha256_hash (keep first occurrence)
before = len(df)
df = df.drop_duplicates(subset=["sha256_hash"])
if PRINT_PREVIEW:
    print(f"Dropped duplicate sha256_hash rows: {before - len(df)}")

# Drop fully duplicate rows (all columns identical)
before = len(df)
df = df.drop_duplicates()
if PRINT_PREVIEW:
    print(f"Fully duplicate rows dropped: {before - len(df)}")

# -------------------------------------------------------
# 7. SIMPLE FEATURE ENGINEERING
# -------------------------------------------------------

# 7.1: VT percent ratio (0.0 - 1.0)
df["vtpercent_ratio"] = df["vtpercent"] / 100.0

# 7.2: time-based features from first_seen_utc
df["first_seen_date"] = df["first_seen_utc"].dt.date
df["first_seen_year"] = df["first_seen_utc"].dt.year
df["first_seen_month"] = df["first_seen_utc"].dt.month
df["first_seen_day"] = df["first_seen_utc"].dt.day

# 7.3: file extension from file_name
def extract_ext(name: str) -> str:
    if not isinstance(name, str) or "." not in name:
        return "no_ext"
    return name.rsplit(".", 1)[-1].lower() or "no_ext"

df["file_ext"] = df["file_name"].apply(extract_ext)

# 7.4: signature / clamav simple flags
df["has_signature"] = df["signature"].notna().astype(int)
df["has_clamav"] = df["clamav"].notna().astype(int)

# -------------------------------------------------------
# 8. LABEL FROM VTPERCENT (FOR ML EXAMPLE)
# -------------------------------------------------------
# vtpercent is "percentage of vendors that detected the file as malicious".
# We make a 3-class label similar to benign/suspicious/malicious.

def vtpercent_to_label(v):
    if pd.isna(v):
        return "unknown"
    v = float(v)
    if v <= 1.0:
        return "benign"
    elif v <= 20.0:
        return "suspicious"
    else:
        return "malicious"

df["label"] = df["vtpercent"].apply(vtpercent_to_label)

label_mapping = {"benign": 0, "suspicious": 1, "malicious": 2, "unknown": 1}
df["label_num"] = df["label"].map(label_mapping).astype(int)

# -------------------------------------------------------
# 9. IMPUTATION & ML-PREP PIPELINE (MEMORY-SAFE)
# -------------------------------------------------------

# Numeric features
numeric_features = [
    "vtpercent",
    "vtpercent_ratio",
    "first_seen_year",
    "first_seen_month",
    "first_seen_day",
    "has_signature",
    "has_clamav",
]
numeric_features = [c for c in numeric_features if c in df.columns]

# Only low/medium-cardinality categoricals for one-hot
categorical_features = [
    "file_type_guess",
    "mime_type",
    "file_ext",
]
categorical_features = [c for c in categorical_features if c in df.columns]

X = df[numeric_features + categorical_features].copy()
y = df["label_num"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    # Use sparse output (default) to avoid giant dense matrices
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# OPTIONAL: sample for prototyping if 1M rows is too big
# sample_size = 200_000
# if len(X) > sample_size:
#     X = X.sample(sample_size, random_state=RANDOM_STATE)
#     y = y.loc[X.index]

X_processed = preprocessor.fit_transform(X)
print("\nProcessed feature matrix shape:", X_processed.shape)

# -------------------------------------------------------
# 10. TRAIN/TEST SPLIT (EXAMPLE)
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Train/test shapes:", X_train.shape, X_test.shape)

# -------------------------------------------------------
# 11. SAVE CLEANED, FEATURE-ENGINEERED DATAFRAME
# -------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)

if PRINT_PREVIEW:
    print("\nFinal cleaned shape (rows, cols):", df.shape)
    print(f"Cleaned CSV written to: {OUTPUT_FILE}")
    print("\nSample of cleaned data:")
    print(df.head())

# -------------------------------------------------------
# 12. OPTIONAL: QUICK VISUAL CHECKS
# -------------------------------------------------------

# Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", order=df["label"].value_counts().index)
plt.title("Label distribution (derived from vtpercent)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# vtpercent distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["vtpercent"].dropna(), bins=40, kde=True)
plt.title("vtpercent distribution")
plt.tight_layout()
plt.show()

# Correlation among numeric features
num_for_corr = ["vtpercent", "vtpercent_ratio", "has_signature", "has_clamav"]
present = [c for c in num_for_corr if c in df.columns]

if len(present) > 1:
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[present].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    plt.show()

print("\nWrangling complete. Output file:", OUTPUT_FILE)
