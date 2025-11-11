# main.py

# --- Imports & display options ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
sns.set(context="notebook", style="whitegrid")

# ==== EDIT THESE PATHS ====
PATH_META = "GSE62944_metadata.csv"  # sample-level metadata (in repo)

# Training expression matrix (original)
PATH_EXPR = "C:/Users/james/OneDrive/Documents/GitHub/Comp_BME_Module_3/Comp_BME_Module_3/GSE62944_subsample_log2TPM.csv"

# External TEST set expression matrix (Module 3 TEST_SET file)
PATH_EXPR_TEST = "C:/Users/james/OneDrive/Documents/GitHub/Comp_BME_Module_3/Comp_BME_Module_3/TEST_SET_GSE62944_subsample_log2TPM.csv"

# Choose your gene set
ANGIO_GENES = ["VEGFA", "HIF1A", "ANGPT2", "FLT1"]  # add KDR, PGF, etc. if you like
TP53_GENE = "TP53"

# --- Sanity check paths ---
if not os.path.exists(PATH_META):
    raise FileNotFoundError(f"Metadata not found: {PATH_META}")
if not os.path.exists(PATH_EXPR):
    raise FileNotFoundError(f"Training expression file not found: {PATH_EXPR}")
if not os.path.exists(PATH_EXPR_TEST):
    raise FileNotFoundError(f"Test expression file not found: {PATH_EXPR_TEST}")

# --- Load metadata & standardize columns ---
meta = pd.read_csv(PATH_META)

colmap = {}
for want, candidates in {
    "sample": ["sample", "sample_id", "SAMPLE", "tcga_sample", "rna_sample"],
    "cancer_type": ["cancer_type", "project", "cancer", "type", "disease"],
    "tissue": ["tissue", "sample_type", "TISSUE", "source_name", "is_tumor"],
}.items():
    for c in candidates:
        if c in meta.columns:
            colmap[want] = c
            break

missing = [k for k in ["sample", "cancer_type"] if k not in colmap]
if missing:
    raise ValueError(
        f"Could not find columns {missing} in metadata. Found columns: {list(meta.columns)}"
    )

meta = meta.rename(
    columns={colmap["sample"]: "sample_id", colmap["cancer_type"]: "cancer_type"}
)
if "tissue" in colmap:
    meta = meta.rename(columns={colmap["tissue"]: "tissue"})

# --- Filter to lung (LUAD + LUSC) ---
lung_meta = meta[meta["cancer_type"].isin(["LUAD", "LUSC"])].copy()
print(lung_meta["cancer_type"].value_counts(dropna=False))
print(f"Lung samples in metadata: {len(lung_meta):,}")

# --- Memory-friendly expression load: read header, keep only lung columns ---
expr_header = pd.read_csv(PATH_EXPR, nrows=0)
expr_cols = expr_header.columns.tolist()
gene_col = expr_cols[0]  # first column should be gene ID / gene symbol
lung_ids = set(lung_meta["sample_id"])

# Keep only gene column + lung sample columns that actually exist in the matrix
keep_cols = [gene_col] + [c for c in expr_cols if c in lung_ids]

if len(keep_cols) <= 1:
    # Helpful debug: show a few examples so you can compare formats
    print("Example lung sample IDs:", list(lung_meta["sample_id"].head(5)))
    print("Example expr columns:", expr_cols[:8])
    raise ValueError(
        "None of the LUAD/LUSC sample IDs matched columns in the expression file.\n"
        "Check that sample_id format matches expression column headers."
    )

print(
    f"Loading expression with {len(keep_cols)-1} lung samples "
    f"out of {len(expr_cols)-1} total columns..."
)
expr = pd.read_csv(PATH_EXPR, usecols=keep_cols).set_index(gene_col)

# Align metadata to loaded expression columns
common_samples = [c for c in expr.columns if c in lung_ids]
lung_expr = expr[common_samples].copy()
lung_meta = lung_meta[lung_meta["sample_id"].isin(common_samples)].copy()

print(f"Expression genes: {lung_expr.shape[0]:,}, lung samples loaded: {lung_expr.shape[1]:,}")
assert set(lung_meta["sample_id"]) == set(lung_expr.columns)

# ============================
# Normalize gene IDs & resolve symbols↔Ensembl for key genes
# ============================
def normalize_gene_index(idx: pd.Index) -> pd.Index:
    s = pd.Index(idx)
    # strip Ensembl version suffix if present (e.g., ENSG... .12)
    s = s.str.replace(r"\.\d+$", "", regex=True)
    # upper-case for consistency
    s = s.str.upper()
    # remove non-alphanumeric/underscore/dash/dot
    s = s.str.replace(r"[^A-Z0-9_\-\.]", "", regex=True)
    return s

lung_expr.index = normalize_gene_index(lung_expr.index)

# Minimal, accurate mapping for our genes of interest
symbol_to_ensembl = {
    "TP53":  "ENSG00000141510",
    "HIF1A": "ENSG00000100644",
    "ANGPT2":"ENSG00000114771",
    "FLT1":  "ENSG00000102755",
    "VEGFA": "ENSG00000112715",
}

def resolve_gene_name(symbol: str, expr_index: pd.Index) -> str | None:
    sym = symbol.upper()
    # 1) try symbol directly
    if sym in expr_index:
        return sym
    # 2) try canonical Ensembl ID
    ens = symbol_to_ensembl.get(sym)
    if ens and ens in expr_index:
        return ens
    # 3) try Ensembl without version (normalize did this already, but just in case)
    if ens and ens.split(".")[0] in expr_index:
        return ens.split(".")[0]
    return None

requested_genes = [TP53_GENE] + ANGIO_GENES
resolved_map = {g: resolve_gene_name(g, lung_expr.index) for g in requested_genes}
missing_after_map = [g for g, r in resolved_map.items() if r is None]
if missing_after_map:
    print("NOTE: these requested genes were not found (symbol or Ensembl):", missing_after_map)

present_pairs = [(g, r) for g, r in resolved_map.items() if r is not None]
if not present_pairs:
    raise ValueError("None of the requested genes were found after normalization/mapping.")

# Build a samples x genes DataFrame with SYMBOL column names
gene_cols = {}
for symbol, row_key in present_pairs:
    gene_cols[symbol] = lung_expr.loc[row_key]
gene_mat = pd.DataFrame(gene_cols)  # index = samples, columns = found genes (by symbol)

# --- Build analysis table (merge metadata) ---
keep_meta_cols = ["sample_id", "cancer_type"]
if "tissue" in lung_meta.columns:
    keep_meta_cols.append("tissue")

df = (
    gene_mat.merge(lung_meta[keep_meta_cols], left_index=True, right_on="sample_id", how="left")
            .set_index("sample_id")
)

# --- Quick sanity check ---
print("df shape:", df.shape)
print("df columns:", list(df.columns))
print("First rows:\n", df.head(3).to_string())

# --- Plots (only use genes that are actually present) ---
plot_vars = [g for g in requested_genes if g in df.columns]
if len(plot_vars) >= 1:
    _ = df[plot_vars].hist(bins=30, figsize=(10, 6))
    plt.tight_layout()
if len(plot_vars) >= 2:
    sns.pairplot(
        df.reset_index(),
        vars=plot_vars,
        hue="cancer_type",
        corner=True,
        plot_kws={"alpha": 0.6, "edgecolor": "k", "linewidth": 0.3},
    )
    plt.show()

# --- Correlations (overall + by subtype) ---
if len(plot_vars) >= 2:
    corr_all = df[plot_vars].corr(method="spearman")
    print("Spearman correlations (lung overall):")
    print(corr_all)

    for ctype in ["LUAD", "LUSC"]:
        sub = df[df["cancer_type"] == ctype]
        if len(sub) >= 10:
            print(f"\nSpearman correlations in {ctype} (n={len(sub)}):")
            print(sub[plot_vars].corr(method="spearman"))
else:
    print("Not enough genes present to compute correlations.")

# ================================
# Classification: LUAD vs LUSC using Logistic Regression
# ================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Select features present in df ---
feature_list = ["TP53", "VEGFA", "HIF1A", "ANGPT2", "FLT1"]
features = [c for c in feature_list if c in df.columns]
if len(features) < 2:
    raise ValueError(f"Need ≥2 features; found {features}")

# --- Build X, y and drop rows with missing values in these columns ---
data_ml = df.dropna(subset=features + ["cancer_type"]).copy()
X = data_ml[features]
y = data_ml["cancer_type"]

# Encode labels -> 0/1
le = LabelEncoder()
y_enc = le.fit_transform(y)   # LUAD/LUSC -> 0/1

# --- Train/test split (stratified) ---
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# --- Pipeline: scale -> logistic regression (elastic-net off for simplicity here) ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=5000, n_jobs=-1
    ))
])

# Fit
pipe.fit(X_tr, y_tr)

# Predict
y_pred = pipe.predict(X_te)
# Prob scores for curves
try:
    y_score = pipe.predict_proba(X_te)[:, 1]
except AttributeError:
    y_score = pipe.decision_function(X_te)

# --- Metrics ---
acc  = accuracy_score(y_te, y_pred)
f1   = f1_score(y_te, y_pred)
auc  = roc_auc_score(y_te, y_score)
aupr = average_precision_score(y_te, y_score)

print("\n=== Logistic Regression (5-gene panel) — Test set ===")
print(f"Accuracy: {acc:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC AUC : {auc:.3f}")
print(f"PR  AUC : {aupr:.3f}")

# --- Plots: ROC, PR, Confusion Matrix ---
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
RocCurveDisplay.from_predictions(y_te, y_score, name="LogReg", ax=ax[0])
ax[0].set_title("ROC Curve")

PrecisionRecallDisplay.from_predictions(y_te, y_score, name="LogReg", ax=ax[1])
ax[1].set_title("Precision–Recall Curve")

ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, normalize="true", display_labels=le.classes_, ax=ax[2]
)
ax[2].set_title("Confusion Matrix (normalized)")
plt.tight_layout()
plt.show()

# --- Coefficient inspection (interpretability) ---
# Recover fitted scaler + coefficients
scaler = pipe.named_steps["scaler"]
clf    = pipe.named_steps["clf"]

coef = clf.coef_[0]
# Because we scaled, coefficients are comparable across features
coef_tbl = pd.DataFrame({"feature": features, "coef": coef, "abs_coef": np.abs(coef)})
coef_tbl = coef_tbl.sort_values("abs_coef", ascending=False)

print("\nTop features by |coefficient|:")
print(coef_tbl[["feature","coef"]].to_string(index=False))

# Plot top coefficients
top_n = min(10, len(features))
top = coef_tbl.head(top_n).sort_values("coef")  # sort for nice barh
plt.figure(figsize=(7, 5))
plt.barh(top["feature"], top["coef"])
plt.xlabel("Coefficient (standardized features)")
plt.title("Logistic Regression — Top Coefficients")
plt.tight_layout()
plt.show()

# --- Optional linear model (VEGFA ~ TP53 + subtype), only if both present ---
try:
    import statsmodels.api as sm
    if all(g in df.columns for g in ["VEGFA", "TP53"]):
        d = df.copy()
        d["is_LUSC"] = (d["cancer_type"] == "LUSC").astype(int)

        def lm_predictor_outcome(outcome, predictor, covars=None, data=None):
            cols = [predictor] + (covars or [])
            d_ = data.dropna(subset=[outcome] + cols).copy()
            X = sm.add_constant(d_[cols])
            y = d_[outcome]
            return sm.OLS(y, X).fit()

        m = lm_predictor_outcome(outcome="VEGFA", predictor="TP53", covars=["is_LUSC"], data=d)
        print(m.summary())
    else:
        print("Skipping linear model: need both VEGFA and TP53 present.")
except Exception as e:
    print("Linear model step skipped:", e)

# --- Add patient_id and save CSV ---
def to_patient_id(sample_barcode):
    if isinstance(sample_barcode, str) and len(sample_barcode) >= 12:
        return sample_barcode[:12]
    return np.nan

df["patient_id"] = [to_patient_id(s) for s in df.index]

out_cols = (["tissue"] if "tissue" in df.columns else []) + ["cancer_type"]
out_cols += [g for g in ["TP53", "VEGFA", "HIF1A", "ANGPT2", "FLT1"] if g in df.columns]
if "patient_id" in df.columns:
    out_cols = ["patient_id"] + out_cols

df[out_cols].to_csv("lung_TP53_angiogenesis_ready.csv")
print("Saved: lung_TP53_angiogenesis_ready.csv")

# ==== COMPLETE, SELF-CONTAINED COMPARISON + ROC (no external deps) ====

# Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay,
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1) Build X, y from df safely (in case some genes were missing earlier)
gene_panel = [g for g in ["TP53","VEGFA","HIF1A","ANGPT2","FLT1"] if g in df.columns]
if len(gene_panel) < 2:
    raise ValueError(f"Need ≥2 genes for modeling, found: {gene_panel}")

X = df[gene_panel].values
# Make LUSC = 1 (positive), LUAD = 0 (negative)
y = (df["cancer_type"].values == "LUSC").astype(int)

# 2) Train/val split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Define three models (Elastic-Net LogReg, GradBoost, RandomForest)
logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(score_func=f_classif, k=min(300, X_train.shape[1]))),
    ("clf", LogisticRegression(
        penalty="elasticnet", solver="saga",
        max_iter=5000, random_state=42
    )),
])

logreg_param = {
    "clf__C": [0.1, 0.3, 1.0, 3.0],
    "clf__l1_ratio": [0.2, 0.5, 0.8],
    # you can keep k fixed since we only have 5 features; left here for completeness
    "kbest__k": [X_train.shape[1]]
}

gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(score_func=f_classif, k=min(300, X_train.shape[1]))),
    ("clf", GradientBoostingClassifier(random_state=42))
])
gb_param = {
    "clf__n_estimators": [150, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [2, 3],
    "kbest__k": [X_train.shape[1]]
}

rf_pipe = Pipeline([
    ("scaler", StandardScaler()),                 # harmless for tree models
    ("kbest", SelectKBest(score_func=f_classif, k=min(300, X_train.shape[1]))),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=1  # n_jobs=1 avoids macOS warnings
    ))
])
# With only 5 genes, RF grid can be tiny or omitted
rf_param = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [None, 3, 5],
    "kbest__k": [X_train.shape[1]]
}

# 4) Fit with small GridSearchCV (n_jobs=1 to avoid child-process noise on macOS)
def fit_model(name, pipe, param_grid):
    gs = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring="roc_auc", cv=5, n_jobs=1, verbose=0
    )
    gs.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print(f"Best CV ROC AUC: {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")
    return gs.best_estimator_

best_logreg     = fit_model("Elastic-Net Logistic Regression", logreg_pipe, logreg_param)
best_gradboost  = fit_model("Gradient Boosting", gb_pipe, gb_param)
best_rf         = fit_model("Random Forest", rf_pipe, rf_param)

# 5) Evaluate on the test set
def eval_model(name, model):
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    y_pred  = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    print(f"\n=== {name} — Test set ===")
    print(f"Accuracy: {acc:.3f}\nF1-score: {f1:.3f}\nROC AUC : {auc:.3f}\nPR  AUC : {ap:.3f}")
    return y_proba, y_pred

proba_logreg,  _ = eval_model("Elastic-Net LogReg", best_logreg)
proba_gb,      _ = eval_model("GradBoost", best_gradboost)
proba_rf,      _ = eval_model("RandomForest", best_rf)

# 6) Plot all ROC curves together
plt.figure(figsize=(7,6))
RocCurveDisplay.from_predictions(y_test, proba_logreg, name="ElasticNet")
RocCurveDisplay.from_predictions(y_test, proba_gb,     name="GradBoost")
RocCurveDisplay.from_predictions(y_test, proba_rf,     name="RandomForest")
plt.title("Model Comparison — ROC Curves")
plt.tight_layout()
plt.show()

# 7) (Optional) Confusion matrix for the winner (change model as you like)
winner = best_gradboost  # or best_logreg / best_rf
y_pred = winner.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(4.5,4))
sns.heatmap(cm, annot=True, cmap="viridis", cbar=True,
            xticklabels=["LUAD","LUSC"], yticklabels=["LUAD","LUSC"],
            fmt=".2f")
plt.title("Confusion Matrix (normalized)")
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.tight_layout(); plt.show()

# ================================
# Supervised task: LUAD vs LUSC (genome-wide/top-k with CV)
# using external TEST_SET expression matrix
# ================================
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)

# ---------- Load & align TEST set ----------
# We already have:
#   - meta, lung_meta (training metadata)
#   - lung_expr (training expression, genes x samples)
#   - normalize_gene_index()

# Load header just to get columns
test_header = pd.read_csv(PATH_EXPR_TEST, nrows=0)
test_cols = test_header.columns.tolist()
gene_col_test = test_cols[0]  # gene ID column

# lung_ids were defined earlier as ALL LUAD/LUSC sample_ids from metadata
# (before filtering to expression)
test_keep_cols = [gene_col_test] + [c for c in test_cols if c in lung_ids]

if len(test_keep_cols) <= 1:
    print("Example lung sample IDs:", list(lung_meta["sample_id"].head(5)))
    print("Example TEST expr columns:", test_cols[:8])
    raise ValueError(
        "None of the LUAD/LUSC sample IDs matched columns in the TEST_SET file.\n"
        "Check that sample_id format matches TEST_SET column headers."
    )

print(
    f"Loading TEST expression with {len(test_keep_cols)-1} lung samples "
    f"out of {len(test_cols)-1} total columns..."
)
test_expr = pd.read_csv(PATH_EXPR_TEST, usecols=test_keep_cols).set_index(gene_col_test)
test_expr.index = normalize_gene_index(test_expr.index)

# Test metadata: LUAD/LUSC samples that appear in the test expression matrix
test_samples = [c for c in test_expr.columns if c in lung_ids]
test_meta = meta[meta["sample_id"].isin(test_samples) & meta["cancer_type"].isin(["LUAD", "LUSC"])].copy()

print(f"TEST_SET genes: {test_expr.shape[0]:,}, lung samples: {test_expr.shape[1]:,}")

# ---------- Build matched train/test matrices (samples x genes) ----------
# Make sure we only use genes present in BOTH train and test
common_genes = lung_expr.index.intersection(test_expr.index)
print(f"Common genes between train and test: {len(common_genes):,}")

X_tr = lung_expr.loc[common_genes].T   # samples x genes
X_te = test_expr.loc[common_genes].T   # samples x genes

# Align labels
y_tr = lung_meta.set_index("sample_id").loc[X_tr.index, "cancer_type"]
y_te = test_meta.set_index("sample_id").loc[X_te.index, "cancer_type"]

print("Training label counts:\n", y_tr.value_counts())
print("TEST label counts:\n", y_te.value_counts())

# Encode LUAD/LUSC -> 0/1
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr)
y_te_enc = le.transform(y_te)
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# ---------- CV setup on TRAINING data only ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Elastic-Net Logistic Regression
logit_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("kbest", SelectKBest(score_func=mutual_info_classif, k=500)),
    ("clf", LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=5000,
        class_weight="balanced", n_jobs=-1
    ))
])

# Model 2: Gradient Boosting
gb_pipe = Pipeline([
    ("kbest", SelectKBest(score_func=mutual_info_classif, k=500)),
    ("clf", GradientBoostingClassifier(random_state=42))
])

# Make sure kbest__k never exceeds number of genes
n_feats = X_tr.shape[1]
logit_k_options = [k for k in [100, 300, 500, 1000] if k <= n_feats]
gb_k_options    = [k for k in [100, 300, 500] if k <= n_feats]

logit_grid = {
    "kbest__k": logit_k_options,
    "clf__C": [0.1, 1.0, 3.0],
    "clf__l1_ratio": [0.0, 0.2, 0.5, 0.8],  # 0=ridge-ish, 1=lasso-ish
}

gb_grid = {
    "kbest__k": gb_k_options,
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [2, 3],
}

# ---------- Fit on TRAINING set (with 5-fold CV) ----------
logit_search = GridSearchCV(
    estimator=logit_pipe, param_grid=logit_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, refit=True
)
logit_search.fit(X_tr, y_tr_enc)
print("\n=== Elastic-Net Logistic Regression (TRAIN CV) ===")
print("Best CV ROC AUC:", round(logit_search.best_score_, 4))
print("Best params:", logit_search.best_params_)

gb_search = GridSearchCV(
    estimator=gb_pipe, param_grid=gb_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, refit=True
)
gb_search.fit(X_tr, y_tr_enc)
print("\n=== Gradient Boosting (TRAIN CV) ===")
print("Best CV ROC AUC:", round(gb_search.best_score_, 4))
print("Best params:", gb_search.best_params_)

# ---------- Pick winner by CV AUC ----------
candidates = [
    ("ElasticNet-LogReg", logit_search),
    ("GradBoost", gb_search),
]
winner_name, winner_search = max(candidates, key=lambda t: t[1].best_score_)
winner = winner_search.best_estimator_
print(f"\n>>> Winner by TRAIN CV AUC: {winner_name} (AUC={winner_search.best_score_:.4f})")

# ---------- FINAL EVALUATION on EXTERNAL TEST_SET ----------
if hasattr(winner, "predict_proba"):
    y_score = winner.predict_proba(X_te)[:, 1]
else:
    y_score = winner.decision_function(X_te)
y_pred = winner.predict(X_te)

test_auc  = roc_auc_score(y_te_enc, y_score)
test_aupr = average_precision_score(y_te_enc, y_score)
test_acc  = accuracy_score(y_te_enc, y_pred)
test_f1   = f1_score(y_te_enc, y_pred)

print("\n=== External TEST_SET metrics ===")
print(f"ROC AUC: {test_auc:.4f}")
print(f"PR AUC:  {test_aupr:.4f}")
print(f"Accuracy:{test_acc:.4f}")
print(f"F1:      {test_f1:.4f}")

# ---------- Plots for TEST_SET ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

RocCurveDisplay.from_predictions(y_te_enc, y_score, name=winner_name, ax=ax[0])
ax[0].set_title("ROC Curve (External TEST_SET)")

PrecisionRecallDisplay.from_predictions(y_te_enc, y_score, name=winner_name, ax=ax[1])
ax[1].set_title("Precision-Recall Curve (External TEST_SET)")

ConfusionMatrixDisplay.from_predictions(
    y_te_enc, y_pred, normalize="true",
    display_labels=le.classes_, ax=ax[2]
)
ax[2].set_title("Confusion Matrix (normalized, External TEST_SET)")
plt.tight_layout()
plt.show()
