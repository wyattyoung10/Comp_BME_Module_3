# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% Load the data
####################################################
data = pd.read_csv(
    '../Data/GSE62944_subsample_topVar_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    '../Data/GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

# %% Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %% Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %% Subset the data for a specific cancer type
####################################################
cancer_type = 'BRCA'  # Breast Invasive Carcinoma

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
BRCA_data = data[cancer_samples]

# %% Subset by index (genes)
####################################################
desired_gene_list = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'MYC']
gene_list = [gene for gene in desired_gene_list if gene in BRCA_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
BRCA_gene_data = BRCA_data.loc[gene_list]
print(BRCA_gene_data.head())

# %% Basic statistics on the subsetted data
####################################################
print(BRCA_gene_data.describe())
print(BRCA_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(BRCA_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(BRCA_gene_data.median(axis=1))

# %% Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_initial_pathologic_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_initial_pathologic_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_initial_pathologic_diagnosis"].mean())
# %% Merging datasets
####################################################
# Merge the subsetted expression data with metadata for BRCA samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
BRCA_metadata = metadata_df.loc[cancer_samples]
BRCA_merged = BRCA_gene_data.T.merge(
    BRCA_metadata, left_index=True, right_index=True)
print(BRCA_merged.head())

# %% Plotting
####################################################
# Boxplot of EGFR expression in BRCA samples using SEABORN
# Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
sns.boxplot(data=BRCA_merged, x="gender", y='EGFR')
plt.title("EGFR Expression by Gender in BRCA Samples")
plt.show()

# Boxplot of MYC and EGFR expression in BRCA samples using PANDAS directly
BRCA_merged[['MYC', 'EGFR']].plot.box()
plt.title("MYC and EGFR Expression in BRCA Samples")
plt.show()

# %% Look a survival characteristics
survival_data = pd.read_excel(
    "../data/Metadata_with_survival.xlsx", index_col=1, header=0, sheet_name='TCGA-CDR')
print(survival_data.head())

# %%
BRCA_survival = survival_data.loc[cancer_samples]
BRCA_merged_survival = BRCA_merged.merge(
    BRCA_survival, left_index=True, right_index=True)
print(BRCA_merged_survival.head())

# %%
