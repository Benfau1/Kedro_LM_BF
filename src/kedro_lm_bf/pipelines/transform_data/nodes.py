import pandas as pd
import numpy as np

def replace_alphanumeric_values(data: pd.DataFrame) -> pd.DataFrame:
    # This function replaces alphanumeric values in the dataframe by interpolating numeric values.
    # It separates columns containing 'before' and 'after' in their names, processes them separately,
    # and then recombines them.
    
    data_cleaned = data.copy()

    # Identify columns related to 'before' and 'after' states
    before_cols = [col for col in data_cleaned.columns if "before" in col]
    after_cols = [col for col in data_cleaned.columns if "after" in col]

    def interpolate_group(df):
        # Convert all values to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Interpolate missing values linearly across columns
        df = df.interpolate(method="linear", axis=1, limit_direction="both")
        # Fill remaining NaNs with the mean of each row
        df = df.fillna(df.mean(axis=1), axis=0)

        # Round and convert to nullable integer type
        return df.round().astype(pd.Int64Dtype())

    # Apply interpolation to 'before' and 'after' column groups separately
    before_cleaned = interpolate_group(data_cleaned[before_cols])
    after_cleaned = interpolate_group(data_cleaned[after_cols])

    # Concatenate the cleaned 'before' and 'after' dataframes side by side
    cleaned_data = pd.concat([before_cleaned, after_cleaned], axis=1)
    # Reorder columns to match the original dataframe's column order
    cleaned_data = cleaned_data[data.columns]
    return cleaned_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # This function cleans the data by converting it to numeric, removing outliers,
    # and interpolating missing values to handle gaps caused by outlier removal.
    
    # Convert all data to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    # Mask values greater than 200 as NaN (considered invalid/outliers)
    data = data.mask(data > 200, np.nan)
    
    # Split columns into two groups: 'before' and 'after' based on the midpoint index
    mid_index = data.shape[1] // 2
    before_cols = data.columns[:mid_index]
    after_cols = data.columns[mid_index:]

    def clean_group(df):
        df = df.copy()

        # Detect outliers column-wise using the IQR method
        Q1_col = df.quantile(0.25)
        Q3_col = df.quantile(0.75)
        IQR_col = Q3_col - Q1_col
        mask_col = (df < (Q1_col - 1.5 * IQR_col)) | (df > (Q3_col + 1.5 * IQR_col))

        # Detect outliers row-wise using the IQR method
        Q1_row = df.quantile(0.25, axis=1)
        Q3_row = df.quantile(0.75, axis=1)
        IQR_row = Q3_row - Q1_row
        mask_row = (df.T < (Q1_row - 1.5 * IQR_row)).T | (df.T > (Q3_row + 1.5 * IQR_row)).T

        # Combine column-wise and row-wise outlier masks
        outliers_mask = mask_col | mask_row
        # Mask outliers as NaN
        df_cleaned = df.mask(outliers_mask, np.nan)

        # Interpolate missing values linearly across columns
        df_cleaned = df_cleaned.interpolate(method="linear", axis=1, limit_direction="both")
        # Fill remaining NaNs with the mean of each row
        df_cleaned = df_cleaned.fillna(df_cleaned.mean(axis=1), axis=0)

        return df_cleaned

    # Clean 'before' and 'after' groups separately
    before_cleaned = clean_group(data[before_cols])
    after_cleaned = clean_group(data[after_cols])

    # Align both cleaned groups on their indices to ensure they match
    before_cleaned, after_cleaned = before_cleaned.align(after_cleaned, axis=0, copy=False)
    # Concatenate the cleaned groups side by side
    data_final = pd.concat([before_cleaned, after_cleaned], axis=1)
    
    # Round and convert to nullable integer type before returning
    return data_final.round().astype(pd.Int64Dtype())