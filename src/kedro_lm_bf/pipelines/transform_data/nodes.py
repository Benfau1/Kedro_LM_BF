import pandas as pd
import numpy as np

def replace_alphanumeric_values(data: pd.DataFrame) -> pd.DataFrame:
    data_cleaned = data.copy()

    before_cols = [col for col in data_cleaned.columns if "before" in col]
    after_cols = [col for col in data_cleaned.columns if "after" in col]

    def interpolate_group(df):
        df = df.apply(pd.to_numeric, errors='coerce')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Interpolation + moyenne par ligne pour combler les extrêmes ou cas limites
        df = df.interpolate(method="linear", axis=1, limit_direction="both")
        df = df.fillna(df.mean(axis=1), axis=0)

        return df.round().astype(pd.Int64Dtype())

    before_cleaned = interpolate_group(data_cleaned[before_cols])
    after_cleaned = interpolate_group(data_cleaned[after_cols])

    cleaned_data = pd.concat([before_cleaned, after_cleaned], axis=1)
    cleaned_data = cleaned_data[data.columns]
    return cleaned_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.mask(data > 200, np.nan)
    

    # Séparation des colonnes
    mid_index = data.shape[1] // 2
    before_cols = data.columns[:mid_index]
    after_cols = data.columns[mid_index:]

    def clean_group(df):
        df = df.copy()

        # --- Outliers colonne par colonne ---
        Q1_col = df.quantile(0.25)
        Q3_col = df.quantile(0.75)
        IQR_col = Q3_col - Q1_col
        mask_col = (df < (Q1_col - 1.5 * IQR_col)) | (df > (Q3_col + 1.5 * IQR_col))

        # --- Outliers ligne par ligne ---
        Q1_row = df.quantile(0.25, axis=1)
        Q3_row = df.quantile(0.75, axis=1)
        IQR_row = Q3_row - Q1_row
        mask_row = (df.T < (Q1_row - 1.5 * IQR_row)).T | (df.T > (Q3_row + 1.5 * IQR_row)).T

        # Combiner les deux masques
        outliers_mask = mask_col | mask_row
        df_cleaned = df.mask(outliers_mask, np.nan)

        # Interpolation
        df_cleaned = df_cleaned.interpolate(method="linear", axis=1, limit_direction="both")
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(axis=1), axis=0)

        return df_cleaned

    # Appliquer par groupe
    before_cleaned = clean_group(data[before_cols])
    after_cleaned = clean_group(data[after_cols])

    before_cleaned, after_cleaned = before_cleaned.align(after_cleaned, axis=0, copy=False)
    data_final = pd.concat([before_cleaned, after_cleaned], axis=1)
    

    return data_final.round().astype(pd.Int64Dtype())