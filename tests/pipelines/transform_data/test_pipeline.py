import pandas as pd
import numpy as np
from kedro_lm_bf.pipelines.transform_data import nodes


def test_replace_alphanumeric_values_handles_letters_and_floats():
    df = pd.DataFrame({
        "before_1": [32, "X", 40],
        "before_2": [26, 27.5, 28],
        "before_3": [24, 25, "A"],
        "before_4": [39, 38, 37],
        "before_5": [20, "B", 22],
        "before_6": [40, 41.0, 42],
        "before_7": [34, 35, np.nan],
        "after_1": [86, 87, "C"],
        "after_2": [84, "N", 85],
        "after_3": [50, 52, 53],
        "after_4": [119, 118, 117],
        "after_5": [115, 9999, 113],
        "after_6": [58, 59, 60],
        "after_7": [76, 77, 78],
    })

    result = nodes.replace_alphanumeric_values(df)

    print("\nRésultat après remplacement alphanumérique :\n", result)

    assert isinstance(result, pd.DataFrame)
    assert not result.isna().any().any(), "Des NaN subsistent après interpolation"
    assert result.dtypes.unique()[0] == pd.Int64Dtype(), "Le type n'est pas Int64"


def test_clean_data_removes_outliers_and_interpolates():
    df = pd.DataFrame({
        "before_1": [32, 1000, 40],
        "before_2": [26, 27, 28],
        "before_3": [24, 25, 26],
        "before_4": [39, 38, 37],
        "before_5": [20, 19, 22],
        "before_6": [40, 41, 42],
        "before_7": [34, 35, 36],
        "after_1": [86, 87, 88],
        "after_2": [84, 85, 86],
        "after_3": [50, 51, 52],
        "after_4": [119, 120, 121],
        "after_5": [115, 116, 9999],
        "after_6": [58, 59, 60],
        "after_7": [76, 77, 78],
    })

    result = nodes.clean_data(df)

    print("\nRésultat après nettoyage (outliers):\n", result)

    assert isinstance(result, pd.DataFrame)
    assert not result.isna().any().any(), "Des NaN subsistent après interpolation"
    assert result.loc[1, "before_1"] != 1000, "Outlier non nettoyé (before_1)"
    assert result.loc[2, "after_5"] != 9999, "Outlier non nettoyé (after_5)"


def test_clean_data_preserves_shape():
    df = pd.DataFrame({
        f"before_{i+1}": [i, i+1, i+2] for i in range(7)
    } | {
        f"after_{i+1}": [i+10, i+11, i+12] for i in range(7)
    })

    result = nodes.clean_data(df)
    assert result.shape == df.shape, "La forme du DataFrame a été modifiée"


def test_replace_alphanumeric_values_preserves_order():
    df = pd.DataFrame({
        "before_1": [1, "X", 3],
        "before_2": [2, 2, 2],
        "after_1": [3, 3, "N"],
        "after_2": [4, 3, 4],
        "before_3": [5, 6, 7],
        "after_3": [8, 9, 10],
        "before_4": [11, 12, 13],
        "after_4": [14, 15, 16],
        "before_5": [17, 18, 19],
        "after_5": [20, 21, 22],
        "before_6": [23, 24, 25],
        "after_6": [26, 27, 28],
        "before_7": [29, 30, 31],
        "after_7": [32, 33, 34],
    })

    result = nodes.replace_alphanumeric_values(df)
    assert list(result.columns) == list(df.columns), "L'ordre des colonnes a changé"


def test_values_within_realistic_range_after_cleaning():
    df = pd.DataFrame({
        "before_1": [0, 130, 999],
        "before_2": [10, 20, 30],
        "before_3": [40, 50, 60],
        "before_4": [70, 80, 90],
        "before_5": [100, 110, 120],
        "before_6": [130, 128, 127],
        "before_7": [126, 125, 124],
        "after_1": [0, 1, 2],
        "after_2": [3, 4, 5],
        "after_3": [6, 7, 8],
        "after_4": [9, 10, 130],
        "after_5": [130, 129, 128],
        "after_6": [127, 126, 999],
        "after_7": [124, 123, 122],
    })

    result = nodes.clean_data(df)

    print("\nValeurs finales après nettoyage :\n", result)

    assert result.max().max() <= 130, "Des valeurs supérieures à 130 subsistent"
    assert result.min().min() >= 0, "Des valeurs inférieures à 0 subsistent"