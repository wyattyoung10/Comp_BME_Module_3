# This code was written with ChatGPT 5.0
# Perfect—let’s make you a clean, classroom-friendly TCGA pan-cancer subset from GSE62944 using Python, with:
# tumors only
# protein-coding genes only (GENCODE v19 / hg19)
# ≥10 cancer types, ~50–100 samples/type (target ~80)
# compact size (~5–10 MB) by keeping the most variable protein-coding genes
# a tidy metadata CSV with ~80 high-value clinical/phenotype columns
# Below is a single Python script you can drop into an empty folder and run. It downloads the exact files you named (defaulting to the 24-cancer June 1, 2015 TPM to match the 548-variable clinical file), pulls a protein-coding list from GENCODE v19, cleans, stratifies, and writes compressed CSVs.
# -----------------------------------------------------------------------
# GSE62944 pan-cancer subset (protein-coding, clean, stratified, compact)
# -----------------------------------------------------------------------
# Requires: Python 3.9+, pandas, numpy
#   pip install pandas numpy
# %%
import os
import io
import gzip
import json
import re
import math
import tarfile
import urllib.request as ureq
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/Users/smgroves/Library/CloudStorage/GoogleDrive-sarahmaddoxgroves@gmail.com/My Drive/Teaching/BME_2315_F2025/Module 3 - Cancer/Data")
OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# 1) Source URLs (GEO & GENCODE)
# -----------------------------
# GSE62944 (TPM & clinical; 24-cancer update, matches 548-variable clinical file)
CLIN_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt.gz"
CTYPE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSE62944_06_01_15_TCGA_24_CancerType_Samples.txt.gz"
# TPM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz"
TPM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1536nnn/GSM1536837/suppl/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz"
# If you truly need the 20-cancer TPM instead (mismatch with 24-cancer clinical), flip the next line:
USE_20C = False
TPM_URL = TPM_URL if not USE_20C else \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62944/suppl/GSM1536837_01_27_15_TCGA_20.Illumina.tumor_Rsubread_TPM.txt.gz"

# GENCODE v19 (hg19) gene annotation to identify protein-coding genes
# (any v19 GTF mirror is OK; these are stable)
GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"


def fetch(url, dest):
    if not dest.exists():
        print(f"Downloading {url}")
        ureq.urlretrieve(url, dest)
    else:
        print(f"Found {dest.name}")


def read_gz_table(path, sep="\t", index_col=None):
    with gzip.open(path, "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()), sep=sep, index_col=index_col)


# %%
# 2) Download data
# -----------------------------
clin_gz = OUT / Path("raw") / Path(CLIN_URL).name
ctype_gz = OUT / Path("raw") / Path(CTYPE_URL).name
tpm_gz = OUT / Path("raw")/Path(TPM_URL).name
gtf_gz = OUT / Path("raw")/Path(GTF_URL).name

for url, dest in [(CLIN_URL, clin_gz), (CTYPE_URL, ctype_gz), (TPM_URL, tpm_gz), (GTF_URL, gtf_gz)]:
    fetch(url, dest)

print("Done with all files.")

# %%
# 3) Load expression (TPM), clinical, cancer-type
# -----------------------------
# TPM matrix is genes x samples
print("Loading data...")
expr = read_gz_table(tpm_gz, sep="\t", index_col=0)
expr.index.name = "gene"
print("Initial expression shape:", expr.shape)

# Clinical table (548 rows; columns should be sample barcode)
clin = read_gz_table(clin_gz, sep="\t")
# First column is usually sample barcode; make it the index robustly
# if clin.columns[0].lower() not in {"sample", "barcode", "tcga_id"}:
# else:
clin = clin.set_index(clin.columns[0])
clin = clin.transpose()
clin.index.name = "sample"

print("Initial clinical shape:", clin.shape)

# Cancer-type lookups: two columns (sample, cancer_type)
ctype = read_gz_table(ctype_gz, sep="\t")
ctype.columns = ["sample", "cancer_type"] + list(ctype.columns[2:])
ctype = ctype.set_index("sample")
print("Initial cancer type shape:", ctype.shape)

# %%
# 4) Keep tumors only and align samples
# -----------------------------
# The TPM file is tumor-only already; just align to clinical + cancer type
print("Aligning samples...")
common = expr.columns.intersection(clin.index).intersection(ctype.index)
expr = expr.loc[:, common]
clin = clin.loc[common]
ctype = ctype.loc[common]

# %%
# 5) Restrict to protein-coding genes (GENCODE v19)
# -----------------------------
# Parse GTF: keep rows with feature == "gene" and gene_type == "protein_coding"
print("Filtering to protein-coding genes...")
pc_genes = []
with gzip.open(gtf_gz, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9:
            continue
        feature = parts[2]
        if feature != "gene":
            continue
        attrs = parts[8]
        # extract gene_name and gene_type (a.k.a. gene_biotype)
        # e.g., gene_name "TP53"; gene_type "protein_coding"
        m_name = re.search(r'gene_name "([^"]+)"', attrs)
        m_type = re.search(r'gene_type "([^"]+)"', attrs)
        if m_name and m_type and m_type.group(1) == "protein_coding":
            pc_genes.append(m_name.group(1))

pc_set = pd.Index(sorted(set(pc_genes)))
expr = expr.loc[expr.index.intersection(pc_set)]

# %%
# 6) Basic cleaning (student-friendly)
# -----------------------------
# a) filter low-expressed genes: TPM >=1 in >=10% samples
print("Basic filtering...")
keep_gene = (expr >= 1.0).sum(axis=1) >= int(np.ceil(0.10 * expr.shape[1]))
expr = expr.loc[keep_gene]

# b) remove tiny-library samples
lib = expr.sum(axis=0)
expr = expr.loc[:, lib >= 1e4]
clin = clin.loc[expr.columns]
ctype = ctype.loc[expr.columns]

# c) log2(TPM+1)
X = np.log2(expr + 1.0)

print(X.shape, "[genes x samples] after basic cleaning")

# %%
# 7) Stratified subsample across cancer types
#     - ≥10 types
#     - 50–100 per type (target ~80)
# -----------------------------
print("Stratified subsampling...")
TARGET, MIN_N, MAX_N = 80, 50, 100
counts = ctype["cancer_type"].value_counts()
types_order = counts.index.tolist()

take = []
rng = np.random.default_rng(SEED)
for ct in types_order:
    cols = np.where(ctype["cancer_type"].values == ct)[0]
    n_avail = len(cols)
    if n_avail < MIN_N:
        # skip very small types to keep balance
        continue
    k = min(MAX_N, max(MIN_N, TARGET, 0), n_avail)
    pick = rng.choice(cols, size=k, replace=False)
    take.extend(list(ctype.index[pick]))

# Ensure at least 10 cancer types kept
if ctype.loc[take, "cancer_type"].nunique() < 10:
    # fallback: take the 10 largest types only
    take = []
    for ct in counts.head(10).index:
        cols = np.where(ctype["cancer_type"].values == ct)[0]
        n_avail = len(cols)
        k = min(MAX_N, max(MIN_N, TARGET, 0), n_avail)
        pick = rng.choice(cols, size=k, replace=False)
        take.extend(list(ctype.index[pick]))

X = X.loc[:, take]
clin = clin.loc[take]
ctype = ctype.loc[take]

print(X.shape, "[genes x samples] after stratified subsampling")
# %%
# 8) Size control: keep top-variance protein-coding genes
# Goal: ~5–10 MB gzip CSV. With ~800 samples, ~1500–2500 genes works well.
# Adjust automatically to hit ~target bytes.
# -----------------------------
print("Selecting top variable genes...")
top_variance = True


def pick_top_var(df, max_genes):
    v = df.var(axis=1)
    keep = v.sort_values(ascending=False).head(max_genes).index
    return df.loc[keep]


if top_variance:
    # heuristic mapping from sample count to gene count for ~5–10MB gz CSV
    ns = X.shape[1]
    # Start with 2000 genes; tweak by sample count
    g0 = 5000 if ns <= 900 else 3000
    X = pick_top_var(X, g0)

# %%
# 9) Curate ~80 “best” metadata columns for undergrad use
#    We select by regex over common TCGA column names in this file,
#    then keep whichever are present.
# -----------------------------
# Buckets: identifiers, demography, tumor info, stage/grade, survival, sample info, treatment, QC-ish
print("Curating metadata columns...")
wanted_patterns = {
    # IDs
    "sample_id": [r"^sample$", r"^barcode", r"^tcga_id", r"^bcr_patient_barcode"],
    # Basic demographics
    "demographics": [r"age", r"gender|sex", r"race", r"ethnicity"],
    # Tumor labelling
    "tumor_labels": [r"tumor_type|cancer_type|project|disease", r"primary_diagnosis", r"anatomic_neoplasm_subdivision|tumor_site"],
    # Stage & grade (AJCC/TNM)
    "staging": [r"ajcc_pathologic_t", r"ajcc_pathologic_n", r"ajcc_pathologic_m",
                r"ajcc_clinical_t", r"ajcc_clinical_n", r"ajcc_clinical_m",
                r"pathologic_stage|clinical_stage",
                r"tumor_grade|neoplasm_histologic_grade|grade"],
    # Survival / outcome
    "survival": [r"vital_status", r"days_to_death|days_to_last_followup|days_to_last_known_alive",
                 r"os_time|os_status|dfs_time|dfs_status|pfs_time|pfs_status|dss_time|dss_status"],
    # Sample attributes
    "sample": [r"sample_type", r"tissue_source_site|tss", r"plate|portion|analyte|aliquot|center", r"batch", r"platform"],
    # Treatment (if present)
    "therapy": [r"pharmaceutical_therapy|radiation_therapy|treatment", r"response|responder"],
    # Purity / QC (often present in extended phenos)
    "purity_qc": [r"purity|est_immune_fraction|est_stromal_fraction|contamination", r"center|plate|portion"],
}

wanted = []
cols = clin.columns.str.lower()
for plist in wanted_patterns.values():
    for p in plist:
        hits = [c for c in clin.columns if re.search(p, c, flags=re.I)]
        wanted.extend(hits)


# Always include cancer_type and sample as the first columns
meta = clin.copy()
meta.insert(0, "sample", meta.index)
meta.insert(1, "cancer_type", ctype["cancer_type"].values)
na_counts = meta.groupby("cancer_type").apply(lambda g: g.isna().mean())
na_median = na_counts.median(axis=0)
across_types = na_median[na_median < .95].sort_values()  # fewest NAs

# Deduplicate, preserve order, cap at ~80 columns total
seen = set()
wanted_uniq = []
for c in wanted:
    if c not in seen:
        wanted_uniq.append(c)
        seen.add(c)
for c in across_types.index.tolist():
    if c not in seen:
        wanted_uniq.append(c)
        seen.add(c)
# keep if in keeped list and not mostly NA across cancer types
# Trim to ~80 columns if we have more
keep_cols = ["sample", "cancer_type"] + \
    [c for c in wanted_uniq if c not in {"sample", "cancer_type"}]
# if len(keep_cols) > 80:
# keep_cols = keep_cols[:80]
meta = meta.reindex(columns=[c for c in keep_cols if c in meta.columns])
print(meta.shape, "[samples x metadata]  after metadata curation")


# %%
# -----------------------------
# 10) Save
# -----------------------------
print("Saving output...")
# X.to_csv(OUT / "GSE62944_subsample_log2TPM.csv.gz", compression="gzip")
# meta.to_csv(OUT / "GSE62944_metadata.csv.gz", index=False, compression="gzip")
if top_variance:
    X.to_csv(OUT / "GSE62944_subsample_topVar_log2TPM.csv")
else:
    X.to_csv(OUT / "GSE62944_subsample_log2TPM.csv")

# %%
meta.to_csv(OUT / "GSE62944_metadata.csv", index=False)
print("Done.")
print("Expression:", X.shape, "→", OUT / "GSE62944_subsample_log2TPM.csv")
print("Metadata:  ", meta.shape, "→", OUT / "GSE62944_metadata.csv")


# %%
# print a summary of cancer types and counts
notna_counts = meta.groupby("cancer_type").apply(lambda g: g.notna().mean())
# Mapping dictionary: TCGA abbreviations to full names
cancer_map = {
    "ACC": "Adrenocortical Carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "CESC": "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon Adenocarcinoma",
    "DLBC": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    "ESCA": "Esophageal Carcinoma",
    "GBM": "Glioblastoma Multiforme",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG": "Brain Lower Grade Glioma",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian Serous Cystadenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate Adenocarcinoma",
    "READ": "Rectum Adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach Adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THCA": "Thyroid Carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UVM": "Uveal Melanoma"
}

# Add full name column to DataFrame
notna_counts["cancer_type_full"] = notna_counts.index.map(cancer_map)
notna_counts.to_csv(OUT / "GSE62944_metadata_percent_nonNA_by_cancer_type.csv")

# %%
X = pd.read_csv(OUT / "GSE62944_subsample_topVar_log2TPM.csv", index_col=0)

# %%
