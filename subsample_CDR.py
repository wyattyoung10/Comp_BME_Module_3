
# %%
import pandas as pd

clin_df = pd.read_csv(
    '../Data/raw/GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt', index_col=0, header=0, sep='\t')

clin_df = clin_df.transpose()
print(clin_df.head())
# %%
survival_data = pd.read_excel(
    "../data/Metadata_with_survival.xlsx", index_col=1, header=0, sheet_name='TCGA-CDR')
print(survival_data.head())

# %%
data = pd.read_csv('../Data/GSE62944_metadata.csv',
                   index_col=0, header=0)
print(data.head())
# %%
subset_survival_data = pd.DataFrame(columns=survival_data.columns)
for i, r in data.iterrows():
    barcode = r['bcr_patient_barcode']
    if barcode in survival_data.index:

        new_row_df = pd.DataFrame(
            [survival_data.loc[barcode]])  # wrap dict in a list
        new_row_df.index = [i]
        subset_survival_data = pd.concat(
            [subset_survival_data, new_row_df], ignore_index=False)
    else:
        print(f"Barcode {i} not found in survival data.")
        # add a row of NaNs
        new_row_df = pd.DataFrame(
            [pd.Series({col: pd.NA for col in survival_data.columns}, name=i)])
        subset_survival_data = pd.concat(
            [subset_survival_data, new_row_df], ignore_index=False)

subset_survival_data = subset_survival_data.drop("Unnamed: 0", axis=1)

subset_survival_data.to_csv('../Data/subsampled_TCGA_CDR_metadata.csv')

# %%
