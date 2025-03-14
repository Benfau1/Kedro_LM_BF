from kedro.pipeline import node
from kedro_lm_bf.data.audiogram_generator import run_generation
import pandas as pd
import numpy as np
import os

def generate_audiogram_data() -> pd.DataFrame:
    """Génère les audiogrammes et retourne un DataFrame."""
    output_file = "data/tonal_exams.csv"

    df = run_generation(10000, output_file)  # Nombre d'audiogrammes à générer

    return df  # On retourne le DataFrame pour Kedro

def replace_alphanumeric_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs non numériques (NaN, lettres) et les valeurs flottantes par 
    une interpolation linéaire basée sur les valeurs numériques voisines, en séparant
    les groupes "before" et "after".
    
    Args:
        data (pd.DataFrame): Jeu de données brut.
    
    Returns:
        pd.DataFrame: Jeu de données nettoyé avec interpolation linéaire.
    """
    data_cleaned = data.copy()

    # Séparer les groupes "before" et "after"
    before_cols = [col for col in data_cleaned.columns if "before" in col]
    after_cols = [col for col in data_cleaned.columns if "after" in col]

    def interpolate_group(df):
        df = df.apply(pd.to_numeric, errors='coerce')
        df.interpolate(method="linear", axis=1, inplace=True, limit_direction="both")
        return df.round().astype(pd.Int64Dtype())  # Conversion en entier avec gestion des NaN

    # Traitement séparé des groupes
    before_cleaned = interpolate_group(data_cleaned[before_cols])
    after_cleaned = interpolate_group(data_cleaned[after_cols])

    # Concaténation des groupes pour reconstituer le DataFrame final
    cleaned_data = pd.concat([before_cleaned, after_cleaned], axis=1)

    return cleaned_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en remplaçant les valeurs aberrantes par une interpolation linéaire par ligne,
    tout en distinguant les groupes "before" et "after".
    """

    data = data.apply(pd.to_numeric, errors='coerce')

    # Identifier les colonnes "before" et "after"
    mid_index = data.shape[1] // 2
    before_cols = data.columns[:mid_index]
    after_cols = data.columns[mid_index:]

    def clean_group(df):
        """Nettoyage d'un sous-groupe de colonnes."""
        df = df.copy()
        
        # Détection des valeurs aberrantes
        Q1 = df.quantile(0.10, axis=1)
        Q3 = df.quantile(0.90, axis=1)
        IQR = Q3 - Q1
        outliers_mask = (df.T < (Q1 - 1.5 * IQR)).T | (df.T > (Q3 + 1.5 * IQR)).T

        # Remplacement des valeurs aberrantes par NaN
        df_cleaned = df.mask(outliers_mask, np.nan)

        # Interpolation linéaire **ligne par ligne**
        df_cleaned = df_cleaned.interpolate(method="linear", axis=1, limit_direction="both")

        return df_cleaned

    # Nettoyage séparé des groupes
    before_cleaned = clean_group(data[before_cols])
    after_cleaned = clean_group(data[after_cols])

    # Assurer l'alignement des colonnes avant la concaténation
    before_cleaned, after_cleaned = before_cleaned.align(after_cleaned, axis=0, copy=False)

    # Concaténation des résultats
    data_final = pd.concat([before_cleaned, after_cleaned], axis=1)

    # Arrondi et conversion en int après interpolation
    data_final = data_final.round().astype(pd.Int64Dtype())

    return data_final