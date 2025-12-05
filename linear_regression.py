import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# --------------------------------------
# 1. LOAD DATA
# --------------------------------------
df = pd.read_csv("cleaned.csv")

print("Loaded cleaned.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# --------------------------------------
# 2. CONVERT TO NUMERIC
#    clamav -> feature (X)
#    vtpercent -> target (y)
# --------------------------------------
df["clamav"] = pd.to_numeric(df["clamav"], errors="coerce")
df["vtpercent"] = pd.to_numeric(df["vtpercent"], errors="coerce")

print("\nNon-NaN counts BEFORE dropping:")
print("clamav    :", df["clamav"].notna().sum())
print("vtpercent :", df["vtpercent"].notna().sum())

# Try strict mode: only rows where BOTH are present
strict_df = df.dropna(subset=["clamav", "vtpercent"])
print(f"\nRows with BOTH clamav and vtpercent present: {len(strict_df)}")

if len(strict_df) >= 10:
    # We have enough real numeric data → use strict_df
    print("Using ONLY rows with real numeric values.")
    df_model = strict_df.copy()
else:
    # No real numeric rows → fall back to filling NaN with 0 (demo mode)
    print("\nWARNING: No (or too few) rows with real numeric clamav/vtpercent.")
    print("Falling back to filling NaN with 0. This is ONLY for demonstration.")
    df_model = df.copy()
    df_model["clamav"] = df_model["clamav"].fillna(0.0)
    df_model["vtpercent"] = df_model["vtpercent"].fillna(0.0)

print("Rows used for regression:", len(df_model))

# Features and target for TRAINING
X = df_model[["clamav"]]
y = df_model["vtpercent"]

# --------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------
if len(df_model) < 2:
    raise SystemExit("Not enough data points to perform train/test split.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTrain samples: {len(y_train)}, Test samples: {len(y_test)}")

# --------------------------------------
# 4. PREPROCESSING + BASELINE LINEAR REGRESSION (PIPELINE)
# --------------------------------------
numeric_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pipe_lr = Pipeline(steps=[
    ("pre", numeric_preprocessor),
    ("lr", LinearRegression())
])

pipe_lr.fit(X_train, y_train)

# --------------------------------------
# 5. EVALUATE BASELINE MODEL
# --------------------------------------
y_pred_lr = pipe_lr.predict(X_test)

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

metrics_lr = regression_metrics(y_test, y_pred_lr)

print("\nBaseline Linear Regression Performance:")
print("---------------------------------------")
print("RMSE:", metrics_lr["RMSE"])
print("MAE :", metrics_lr["MAE"])
print("R²  :", metrics_lr["R2"])

# Learned relationship (approx, using raw feature name)
lr_coef = pipe_lr.named_steps["lr"].coef_[0]
lr_intercept = pipe_lr.named_steps["lr"].intercept_
print("\nLearned relationship (on scaled clamav, but shown simply as):")
print(f"vtpercent ≈ {lr_intercept:.4f} + {lr_coef:.4f} * clamav_scaled")

# Plot predicted vs actual (baseline)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.4)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual vtpercent")
plt.ylabel("Predicted vtpercent (LR)")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------
# 6. CROSS-VALIDATION FOR BASELINE LR
# --------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
neg_mse_scores = cross_val_score(
    pipe_lr, X, y,
    cv=cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)
rmse_scores = np.sqrt(-neg_mse_scores)
print("\nCross-validated RMSE (5-fold, Linear Regression):")
print("Mean RMSE:", rmse_scores.mean())
print("Std  RMSE:", rmse_scores.std())

# --------------------------------------
# 7. REGULARIZED MODELS: RIDGE & LASSO WITH GRIDSEARCHCV
# --------------------------------------
alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

pipe_ridge = Pipeline(steps=[
    ("pre", numeric_preprocessor),
    ("ridge", Ridge(random_state=RANDOM_STATE))
])

pipe_lasso = Pipeline(steps=[
    ("pre", numeric_preprocessor),
    ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=5000))
])

param_grid_ridge = {"ridge__alpha": alphas}
param_grid_lasso = {"lasso__alpha": alphas}

gs_ridge = GridSearchCV(
    pipe_ridge,
    param_grid_ridge,
    cv=cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

gs_lasso = GridSearchCV(
    pipe_lasso,
    param_grid_lasso,
    cv=cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

print("\nFitting Ridge...")
gs_ridge.fit(X_train, y_train)
print("Fitting Lasso...")
gs_lasso.fit(X_train, y_train)

best_ridge = gs_ridge.best_estimator_
best_lasso = gs_lasso.best_estimator_

print("\nBest Ridge alpha:", gs_ridge.best_params_, 
      "CV RMSE:", np.sqrt(-gs_ridge.best_score_))
print("Best Lasso alpha:", gs_lasso.best_params_, 
      "CV RMSE:", np.sqrt(-gs_lasso.best_score_))

# Evaluate on test set
y_pred_ridge = best_ridge.predict(X_test)
y_pred_lasso = best_lasso.predict(X_test)

metrics_ridge = regression_metrics(y_test, y_pred_ridge)
metrics_lasso = regression_metrics(y_test, y_pred_lasso)

print("\nRidge Regression Test Performance:")
print("----------------------------------")
print("RMSE:", metrics_ridge["RMSE"])
print("MAE :", metrics_ridge["MAE"])
print("R²  :", metrics_ridge["R2"])

print("\nLasso Regression Test Performance:")
print("----------------------------------")
print("RMSE:", metrics_lasso["RMSE"])
print("MAE :", metrics_lasso["MAE"])
print("R²  :", metrics_lasso["R2"])

# --------------------------------------
# 8. SAVE BASELINE MODEL (PIPELINE)
# --------------------------------------
joblib.dump(pipe_lr, "linear_regression_clamav_vtpercent.pkl")
print("\nBaseline Linear Regression pipeline saved as linear_regression_clamav_vtpercent.pkl")

# --------------------------------------
# 9. PREDICT FOR THE FULL CSV AND SAVE
# --------------------------------------
df_full = pd.read_csv("cleaned.csv")
df_full["clamav"] = pd.to_numeric(df_full["clamav"], errors="coerce")

X_full = df_full[["clamav"]]

# Predict with all three models
vtpred_full_lr = pipe_lr.predict(X_full)
vtpred_full_ridge = best_ridge.predict(X_full)
vtpred_full_lasso = best_lasso.predict(X_full)

df_full["vtpercent_pred_lr"] = vtpred_full_lr
df_full["vtpercent_pred_ridge"] = vtpred_full_ridge
df_full["vtpercent_pred_lasso"] = vtpred_full_lasso

output_file = "cleaned_with_vtpercent_pred.csv"
df_full.to_csv(output_file, index=False)
print(f"\nFull predictions saved to: {output_file}")

# --------------------------------------
# 10. SHOW ONLY FIRST 5 PREDICTIONS
# --------------------------------------
print("\nSample predictions (first 5 rows of full CSV):")
print(df_full[["clamav", "vtpercent_pred_lr", "vtpercent_pred_ridge", "vtpercent_pred_lasso"]].head())

print("\nDone. Baseline and regularized regressions complete.")
