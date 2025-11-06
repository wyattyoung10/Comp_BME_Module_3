# main.py

# --- Imports & display options ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
sns.set(context="notebook", style="whitegrid")

# ==== EDIT THESE PATHS ====
PATH_META = "GSE62944_metadata.csv"                  # in repo
PATH_EXPR = "/Users/wyattyoung/Desktop/GSE62944_subsample_log2TPM.csv"  # on Desktop

ANGIO_GENES = ["VEGFA", "HIF1A", "ANGPT2", "FLT1"]
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
    raise ValueError(f"Could not find columns {missing} in metadata. Found columns: {list(meta.columns)}")

meta = meta.rename(columns={colmap["sample"]: "sample_id", colmap["cancer_type"]: "cancer_type"})
if "tissue" in colmap:
    meta = meta.rename(columns={colmap["tissue"]: "tissue"})

# --- Filter to lung (LUAD + LUSC) ---
lung_meta = meta[meta["cancer_type"].isin(["LUAD", "LUSC"])].copy()
print(lung_meta["cancer_type"].value_counts(dropna=False))
print(f"Lung samples in metadata: {len(lung_meta):,}")

# --- Memory-friendly expression load: keep only lung columns ---
expr_header = pd.read_csv(PATH_EXPR, nrows=0)
expr_cols = expr_header.columns.tolist()
gene_col = expr_cols[0]
lung_ids = set(lung_meta["sample_id"])
keep_cols = [gene_col] + [c for c in expr_cols if c in lung_ids]
if len(keep_cols) <= 1:
    print("Example lung sample IDs:", list(lung_meta["sample_id"].head(5)))
    print("Example expr columns:", expr_cols[:8])
    raise ValueError("None of the LUAD/LUSC sample IDs matched expression columns.")

print(f"Loading expression with {len(keep_cols)-1} lung samples out of {len(expr_cols)-1} total columns...")
expr = pd.read_csv(PATH_EXPR, usecols=keep_cols).set_index(gene_col)

# Align
common_samples = [c for c in expr.columns if c in lung_ids]
lung_expr = expr[common_samples].copy()
lung_meta = lung_meta[lung_meta["sample_id"].isin(common_samples)].copy()
print(f"Expression genes: {lung_expr.shape[0]:,}, lung samples loaded: {lung_expr.shape[1]:,}")

# --- Normalize gene IDs & resolve symbols↔Ensembl for our panel ---
def normalize_gene_index(idx: pd.Index) -> pd.Index:
    s = pd.Index(idx).str.replace(r"\.\d+$", "", regex=True).str.upper()
    s = s.str.replace(r"[^A-Z0-9_\-\.]", "", regex=True)
    return s
lung_expr.index = normalize_gene_index(lung_expr.index)

symbol_to_ensembl = {
    "TP53":  "ENSG00000141510",
    "HIF1A": "ENSG00000100644",
    "ANGPT2":"ENSG00000114771",
    "FLT1":  "ENSG00000102755",
    "VEGFA": "ENSG00000112715",
}
def resolve(symbol: str, idx: pd.Index):
    sym = symbol.upper()
    if sym in idx: return sym
    ens = symbol_to_ensembl.get(sym)
    if ens and ens in idx: return ens
    ens_base = ens.split(".")[0] if ens else None
    if ens_base and ens_base in idx: return ens_base
    return None

requested = [TP53_GENE] + ANGIO_GENES
resolved = {g: resolve(g, lung_expr.index) for g in requested}
missing = [g for g, r in resolved.items() if r is None]
if missing:
    print("NOTE: requested genes not found (symbol or Ensembl):", missing)

present = [(g, r) for g, r in resolved.items() if r is not None]
if not present:
    raise ValueError("None of the requested genes were found.")

gene_mat = pd.DataFrame({sym: lung_expr.loc[rowkey] for sym, rowkey in present})

# --- Build analysis table ---
keep_meta_cols = ["sample_id", "cancer_type"] + (["tissue"] if "tissue" in lung_meta.columns else [])
df = gene_mat.merge(lung_meta[keep_meta_cols], left_index=True, right_on="sample_id", how="left").set_index("sample_id")
print("df shape:", df.shape)
print("df columns:", list(df.columns))
print("First rows:\n", df.head(3).to_string())

# ============================
# Classification + Model comparison (5-gene panel)
# ============================
gene_panel = [g for g in ["TP53","VEGFA","HIF1A","ANGPT2","FLT1"] if g in df.columns]
if len(gene_panel) < 2:
    raise ValueError(f"Need ≥2 genes for modeling, found: {gene_panel}")

X = df[gene_panel].values
y = (df["cancer_type"].values == "LUSC").astype(int)  # LUSC=1, LUAD=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Elastic-Net LogReg
logreg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(score_func=f_classif, k=X_train.shape[1])),
    ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=42))
])
logreg_param = {"clf__C":[0.1, 0.3, 1.0], "clf__l1_ratio":[0.2, 0.5, 0.8], "kbest__k":[X_train.shape[1]]}

# Gradient Boosting
gb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kbest", SelectKBest(score_func=f_classif, k=X_train.shape[1])),
    ("clf", GradientBoostingClassifier(random_state=42))
])
gb_param = {"clf__n_estimators":[150,200], "clf__learning_rate":[0.05,0.1], "clf__max_depth":[2,3], "kbest__k":[X_train.shape[1]]}

# Random Forest
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),                                   # harmless for trees
    ("kbest", SelectKBest(score_func=f_classif, k=X_train.shape[1])),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=1))
])
rf_param = {"clf__n_estimators":[200,300], "clf__max_depth":[None,3,5], "kbest__k":[X_train.shape[1]]}

def fit_model(name, pipe, grid):
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print(f"Best CV ROC AUC: {gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")
    return gs.best_estimator_

best_logreg    = fit_model("Elastic-Net Logistic Regression", logreg_pipe, logreg_param)
best_gradboost = fit_model("Gradient Boosting", gb_pipe, gb_param)
best_rf        = fit_model("Random Forest", rf_pipe, rf_param)

def eval_model(name, model):
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    pred  = (proba >= 0.5).astype(int)
    print(f"\n=== {name} — Test set ===")
    print(f"Accuracy: {accuracy_score(y_test,pred):.3f}")
    print(f"F1-score: {f1_score(y_test,pred):.3f}")
    print(f"ROC AUC : {roc_auc_score(y_test,proba):.3f}")
    print(f"PR  AUC : {average_precision_score(y_test,proba):.3f}")
    return proba, pred

proba_lr, _  = eval_model("Elastic-Net LogReg", best_logreg)
proba_gb, _  = eval_model("GradBoost", best_gradboost)
proba_rf, _  = eval_model("RandomForest", best_rf)

# ROC comparison
plt.figure(figsize=(7,6))
RocCurveDisplay.from_predictions(y_test, proba_lr, name="ElasticNet")
RocCurveDisplay.from_predictions(y_test, proba_gb, name="GradBoost")
RocCurveDisplay.from_predictions(y_test, proba_rf, name="RandomForest")
plt.title("Model Comparison — ROC Curves")
plt.tight_layout(); plt.show()

# Confusion matrix for the winner (pick one)
winner = best_gradboost
y_hat = winner.predict(X_test)
cm = confusion_matrix(y_test, y_hat, normalize="true")
plt.figure(figsize=(4.5,4))
sns.heatmap(cm, annot=True, cmap="viridis", cbar=True,
            xticklabels=["LUAD","LUSC"], yticklabels=["LUAD","LUSC"], fmt=".2f")
plt.title("Confusion Matrix (normalized)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()

# Save CSV with patient_id + values
def to_patient_id(barcode):
    return barcode[:12] if isinstance(barcode, str) and len(barcode) >= 12 else np.nan
df["patient_id"] = [to_patient_id(s) for s in df.index]
cols = (["tissue"] if "tissue" in df.columns else []) + ["cancer_type"] + gene_panel
df[["patient_id"] + cols].to_csv("lung_TP53_angiogenesis_ready.csv", index=True)
print("Saved: lung_TP53_angiogenesis_ready.csv")
