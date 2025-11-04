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
PATH_EXPR = "/Users/wyattyoung/Desktop/GSE62944_subsample_log2TPM.csv"  # expression on Desktop

# Choose your gene set
ANGIO_GENES = ["VEGFA", "HIF1A", "ANGPT2", "FLT1"]  # add KDR, PGF, etc. if you like
TP53_GENE = "TP53"

# --- Sanity check paths ---
if not os.path.exists(PATH_META):
    raise FileNotFoundError(f"Metadata not found: {PATH_META}")
if not os.path.exists(PATH_EXPR):
    raise FileNotFoundError(f"Expression file not found: {PATH_EXPR}")

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

# ================================
# Supervised task: LUAD vs LUSC (genome-wide/top-k with CV)
# ================================
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X = samples x genes; y = labels aligned to X.index
X = lung_expr.T.copy()
y = lung_meta.set_index("sample_id").loc[X.index, "cancer_type"].copy()

# encode LUAD/LUSC -> 0/1
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# Model 1: Elastic-Net Logistic Regression
# ---------------------------
logit_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("kbest", SelectKBest(score_func=mutual_info_classif, k=500)),
    ("clf", LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=5000,
        class_weight="balanced", n_jobs=-1
    ))
])

logit_grid = {
    "kbest__k": [100, 300, 500, 1000],
    "clf__C": [0.1, 1.0, 3.0],
    "clf__l1_ratio": [0.0, 0.2, 0.5, 0.8],  # 0=ridge-ish, 1=lasso-ish
}

logit_search = GridSearchCV(
    estimator=logit_pipe, param_grid=logit_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, refit=True
)
logit_search.fit(X_tr, y_tr)
print("\n=== Elastic-Net Logistic Regression ===")
print("Best CV ROC AUC:", round(logit_search.best_score_, 4))
print("Best params:", logit_search.best_params_)

# ---------------------------
# Model 2: Gradient Boosting
# ---------------------------
gb_pipe = Pipeline([
    ("kbest", SelectKBest(score_func=mutual_info_classif, k=500)),
    ("clf", GradientBoostingClassifier(random_state=42))
])

gb_grid = {
    "kbest__k": [100, 300, 500],
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [2, 3],
}

gb_search = GridSearchCV(
    estimator=gb_pipe, param_grid=gb_grid, cv=cv,
    scoring="roc_auc", n_jobs=-1, refit=True
)
gb_search.fit(X_tr, y_tr)
print("\n=== Gradient Boosting ===")
print("Best CV ROC AUC:", round(gb_search.best_score_, 4))
print("Best params:", gb_search.best_params_)

# ---------------------------
# Pick winner by CV AUC & evaluate on test
# ---------------------------
candidates = [
    ("ElasticNet-LogReg", logit_search),
    ("GradBoost", gb_search),
]
winner_name, winner_search = max(candidates, key=lambda t: t[1].best_score_)
winner = winner_search.best_estimator_
print(f"\n>>> Winner by CV AUC: {winner_name} (AUC={winner_search.best_score_:.4f})")

# Test-set evaluation
if hasattr(winner, "predict_proba"):
    y_score = winner.predict_proba(X_te)[:, 1]
else:
    # fall back to decision_function if no predict_proba
    y_score = winner.decision_function(X_te)
y_pred = winner.predict(X_te)

test_auc = roc_auc_score(y_te, y_score)
test_aupr = average_precision_score(y_te, y_score)
test_acc = accuracy_score(y_te, y_pred)
test_f1  = f1_score(y_te, y_pred)

print("\n=== Test-set metrics ===")
print(f"ROC AUC: {test_auc:.4f}")
print(f"PR AUC:  {test_aupr:.4f}")
print(f"Accuracy:{test_acc:.4f}")
print(f"F1:      {test_f1:.4f}")

# ---------------------------
# Plots: ROC, PR, Confusion Matrix
# ---------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

RocCurveDisplay.from_predictions(y_te, y_score, name=winner_name, ax=ax[0])
ax[0].set_title("ROC Curve")

PrecisionRecallDisplay.from_predictions(y_te, y_score, name=winner_name, ax=ax[1])
ax[1].set_title("Precision-Recall Curve")

ConfusionMatrixDisplay.from_predictions(y_te, y_pred, normalize="true", display_labels=le.classes_, ax=ax[2])
ax[2].set_title("Confusion Matrix (normalized)")
plt.tight_layout()
plt.show()

# ---------------------------
# Feature importance plot (top 20)
# ---------------------------
def plot_top_features(estimator, Xframe, title, top_n=20):
    # Recover selected feature names after SelectKBest
    if "kbest" in estimator.named_steps:
        support_idx = estimator.named_steps["kbest"].get_support(indices=True)
        selected_cols = Xframe.columns[support_idx]
    else:
        selected_cols = Xframe.columns

    if "clf" in estimator.named_steps:
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator  # in case the pipeline is different

    # Logistic: use absolute coefficients
    if hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_[0])
        names = selected_cols

    # Tree-based: use feature_importances_
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        names = selected_cols
    else:
        print("No feature importances/coefficients found for this model.")
        return

    order = np.argsort(vals)[-top_n:][::-1]
    top_names = names[order]
    top_vals = vals[order]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_names)), top_vals[::-1])
    plt.yticks(range(len(top_names)), top_names[::-1], fontsize=8)
    plt.xlabel("Importance (|coef| or impurity)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_top_features(winner, X_tr, f"Top features — {winner_name}", top_n=20)
