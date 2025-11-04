# Import and display options
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
sns.set(context="notebook", style="whitegrid")

# ==== EDIT THESE PATHS ====
PATH_META = "GSE62944_metadata.csv"                # sample-level metadata
PATH_EXPR = "GSE62944_subsample_log2TPM.csv"       # expression matrix (genes x samples, log2 TPM)
# If your expression is zipped as .zip with a CSV inside, unzip first or read with pandas + zipfile.

# Optional survival (if available):
PATH_SURV = "subsampled_TCGA_CDR_survival.csv"     # patient-level survival table (optional)

# Choose your gene set
ANGIO_GENES = ["VEGFA", "HIF1A", "ANGPT2", "FLT1"]  # you can add KDR, PGF, HIF1B, etc.
TP53_GENE = "TP53"

# Load metadata and standardize columns
meta = pd.read_csv(PATH_META)

# Try to infer likely column names (so this works across variants)
colmap = {}
for want, candidates in {
    "sample": ["sample", "sample_id", "SAMPLE", "tcga_sample", "rna_sample"],
    "cancer_type": ["cancer_type", "project", "cancer", "type", "disease"],
    "tissue": ["tissue", "sample_type", "TISSUE", "source_name", "is_tumor"]
}.items():
    for c in candidates:
        if c in meta.columns:
            colmap[want] = c
            break

missing = [k for k in ["sample", "cancer_type"] if k not in colmap]
if missing:
    raise ValueError(f"Could not find columns {missing} in metadata. Found columns: {list(meta.columns)}")

# Normalize names
meta = meta.rename(columns={colmap["sample"]: "sample_id",
                            colmap["cancer_type"]: "cancer_type"})
if "tissue" in colmap:
    meta = meta.rename(columns={colmap["tissue"]: "tissue"})

# Keep only LUAD + LUSC
lung_meta = meta[meta["cancer_type"].isin(["LUAD", "LUSC"])].copy()
print(lung_meta["cancer_type"].value_counts(dropna=False))
print(f"Lung samples in metadata: {len(lung_meta):,}")

# load expression matrix and subset to lung samples
# Expression format expected: rows = genes, columns = sample_ids (matching metadata)
expr = pd.read_csv(PATH_EXPR, index_col=0)

# Intersect columns with lung_meta sample IDs (drop any missing)
common_samples = [s for s in lung_meta["sample_id"] if s in expr.columns]
lung_expr = expr[common_samples].copy()
lung_meta = lung_meta[lung_meta["sample_id"].isin(common_samples)].copy()

print(f"Expression genes: {expr.shape[0]:,}, total samples: {expr.shape[1]:,}")
print(f"Lung subset: genes {lung_expr.shape[0]:,}, samples {lung_expr.shape[1]:,}")
assert set(lung_meta["sample_id"]) == set(lung_expr.columns)


# build an analysis table with genes
genes_needed = [TP53_GENE] + ANGIO_GENES
missing_genes = [g for g in genes_needed if g not in lung_expr.index]
if missing_genes:
    print("WARNING: missing genes in expression:", missing_genes)

present_genes = [g for g in genes_needed if g in lung_expr.index]
wide = lung_expr.loc[present_genes].T  # samples x genes

# Add metadata columns of interest
keep_meta_cols = ["sample_id", "cancer_type"]
if "tissue" in lung_meta.columns:
    keep_meta_cols.append("tissue")

df = wide.merge(lung_meta[keep_meta_cols], left_index=True, right_on="sample_id", how="left")
df = df.set_index("sample_id")
df.head()


# distribution and plots
# Histograms of TP53 and VEGFA/HIF1A
_ = df[[TP53_GENE] + [g for g in ANGIO_GENES if g in df.columns]].hist(bins=30, figsize=(10,6))
plt.tight_layout()

# Pairplot to visually inspect correlation (colored by cancer_type)
sns.pairplot(df.reset_index(), vars=[TP53_GENE] + [g for g in ANGIO_GENES if g in df.columns],
             hue="cancer_type", corner=True, plot_kws={"alpha":0.6, "edgecolor":"k", "linewidth":0.3})
plt.show()


# correlations in lung
# Overall Spearman correlations (robust to nonlinearity/outliers)
genes_for_corr = [TP53_GENE] + [g for g in ANGIO_GENES if g in df.columns]
corr_all = df[genes_for_corr].corr(method="spearman")
print("Spearman correlations (lung overall):")
display(corr_all)

# Correlations stratified by LUAD / LUSC
for ctype in ["LUAD", "LUSC"]:
    sub = df[df["cancer_type"] == ctype]
    if len(sub) >= 10:
        print(f"\nSpearman correlations in {ctype} (n={len(sub)}):")
        display(sub[genes_for_corr].corr(method="spearman"))


# optional linear models
import statsmodels.api as sm

def lm_predictor_outcome(outcome, predictor, covars=None, data=None):
    cols = [predictor]
    if covars:
        cols += covars
    d = data.dropna(subset=[outcome] + cols).copy()
    X = d[[predictor] + (covars or [])]
    X = sm.add_constant(X)
    y = d[outcome]
    model = sm.OLS(y, X).fit()
    return model

# Example: VEGFA ~ TP53 + cancer_type (LUAD/LUSC as dummy)
if "VEGFA" in df.columns:
    d = df.copy()
    d["is_LUSC"] = (d["cancer_type"] == "LUSC").astype(int)
    m = lm_predictor_outcome(outcome="VEGFA", predictor=TP53_GENE, covars=["is_LUSC"], data=d)
    print(m.summary())


# save file
out_cols = ["cancer_type"] + [TP53_GENE] + [g for g in ANGIO_GENES if g in df.columns]
if "tissue" in df.columns:
    out_cols = ["tissue"] + out_cols
out_cols = ["patient_id"] + out_cols if "patient_id" in df.columns else out_cols

df[out_cols].to_csv("lung_TP53_angiogenesis_ready.csv")
print("Saved: lung_TP53_angiogenesis_ready.csv")
