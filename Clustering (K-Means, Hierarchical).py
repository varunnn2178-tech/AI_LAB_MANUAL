# ==============================================================
# Unified Clustering Script: K-Means + Hierarchical + PCA + Dendrogram
# With safe memory settings for Hierarchical
# ==============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
CSV_PATH = "cleaned.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# 2. SEPARATE TARGET (signature)
# ---------------------------------------------------------
TARGET_COL = "signature"
has_signature = TARGET_COL in df.columns

if has_signature:
    y_signature_full = df[TARGET_COL].astype(str)
    X_full = df.drop(columns=[TARGET_COL]).copy()
else:
    X_full = df.copy()
    y_signature_full = None

# ---------------------------------------------------------
# 3. SELECT NUMERIC FEATURES (for reference / design, but we encode all)
# ---------------------------------------------------------
numeric_features = [
    "vtpercent", "vtpercent_ratio",
    "first_seen_year", "first_seen_month",
    "first_seen_day", "has_signature", "has_clamav"
]
numeric_features = [c for c in numeric_features if c in X_full.columns]

tag_features = [c for c in X_full.columns if c.startswith("tag_")]
numeric_features += tag_features

print("\nNumeric features used:", numeric_features)

# ---------------------------------------------------------
# 4. SUBSAMPLE for SPEED (for K-Means & PCA)
# ---------------------------------------------------------
MAX_ROWS = 50000
if len(X_full) > MAX_ROWS:
    print(f"\nSubsampling to {MAX_ROWS} rows for clustering...")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_full), size=MAX_ROWS, replace=False)
    X = X_full.iloc[idx].copy()
    if y_signature_full is not None:
        y_signature = y_signature_full.iloc[idx]
else:
    X = X_full.copy()
    y_signature = y_signature_full
    print(f"\nDataset contains {len(X_full)} rows; no subsampling applied.")

# ---------------------------------------------------------
# 5. ENCODE ALL FEATURES (strings, hashes, etc.)
# ---------------------------------------------------------
feature_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    feature_encoders[col] = le

# ---------------------------------------------------------
# 6. SCALE FEATURES
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Feature Matrix Shape:", X_scaled.shape)

# ---------------------------------------------------------
# 7. PCA (for 2D visualization)
# ---------------------------------------------------------
print("\nRunning PCA for 2D visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------------
# 8. K-MEANS CLUSTERING (on 50K rows)
# ---------------------------------------------------------
K_CLUSTERS = 5
print(f"\nRunning KMeans with {K_CLUSTERS} clusters...")

kmeans = KMeans(
    n_clusters=K_CLUSTERS,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

# Dataframe for KMeans results (subsample only)
cluster_df_kmeans = pd.DataFrame(index=X.index)
cluster_df_kmeans["kmeans_cluster"] = kmeans_labels
if has_signature:
    cluster_df_kmeans["signature"] = y_signature.values

print("\nKMeans cluster counts:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

# ---------- PCA Plot for K-Means ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels,
    palette="Set1", s=25
)
plt.title("K-Means Clusters (PCA 2D Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", loc="upper right")
plt.tight_layout()
plt.show()

# ---------- K-Means Cluster Distribution ----------
plt.figure(figsize=(7, 4))
sns.countplot(x=kmeans_labels, palette="Set1")
plt.title("K-Means Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. HIERARCHICAL CLUSTERING (on smaller subset to avoid MemoryError)
# ---------------------------------------------------------
HIER_MAX_ROWS = 3000   # <= safe for RAM; you can try 2000–5000
print(f"\nPreparing data for Hierarchical Clustering (max {HIER_MAX_ROWS} rows)...")

if X_scaled.shape[0] > HIER_MAX_ROWS:
    rng_h = np.random.RandomState(123)
    hier_idx = rng_h.choice(X_scaled.shape[0], size=HIER_MAX_ROWS, replace=False)
else:
    hier_idx = np.arange(X_scaled.shape[0])

X_hier = X_scaled[hier_idx]
X_pca_hier = X_pca[hier_idx]
if has_signature:
    sig_hier = y_signature.iloc[hier_idx].values
else:
    sig_hier = None

H_CLUSTERS = 5
print(f"\nRunning Hierarchical Clustering with {H_CLUSTERS} clusters on {X_hier.shape[0]} samples...")

hier = AgglomerativeClustering(
    n_clusters=H_CLUSTERS,
    metric="euclidean",
    linkage="ward"
)

hier_labels = hier.fit_predict(X_hier)

# Dataframe for Hierarchical results (subset only)
cluster_df_hier = pd.DataFrame({
    "hier_cluster": hier_labels
})
if sig_hier is not None:
    cluster_df_hier["signature"] = sig_hier

print("\nHierarchical cluster counts:")
print(pd.Series(hier_labels).value_counts().sort_index())

# ---------- PCA Plot for Hierarchical ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca_hier[:, 0], y=X_pca_hier[:, 1], hue=hier_labels,
    palette="Set2", s=25
)
plt.title("Hierarchical Clusters (PCA 2D Projection, subset)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", loc="upper right")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. DENDROGRAM (even smaller subset, e.g., 200 rows)
# ---------------------------------------------------------
DENDRO_ROWS = 200
print(f"\nPlotting dendrogram for first {min(DENDRO_ROWS, X_hier.shape[0])} samples...")

X_dendro = X_hier[:DENDRO_ROWS]
Z = linkage(X_dendro, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="level", p=10)
plt.title("Hierarchical Clustering Dendrogram (subset)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. CLUSTER vs SIGNATURE crosstabs
# ---------------------------------------------------------
if has_signature:
    print("\n[Subsample] KMeans Cluster vs Signature (first 10 signatures):")
    ct_k = pd.crosstab(cluster_df_kmeans["kmeans_cluster"], cluster_df_kmeans["signature"])
    print(ct_k.iloc[:, :10])

    print("\n[Subset] Hierarchical Cluster vs Signature (first 10 signatures):")
    ct_h = pd.crosstab(cluster_df_hier["hier_cluster"], cluster_df_hier["signature"])
    print(ct_h.iloc[:, :10])

# ---------------------------------------------------------
print("\n=== DONE: K-Means + Hierarchical Clustering Completed Successfully ===")
