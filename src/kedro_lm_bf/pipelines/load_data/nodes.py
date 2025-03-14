from kedro.pipeline import node
from kedro_lm_bf.data.audiogram_generator import run_generation
import pandas as pd
import numpy as np
import os

def generate_audiogram_data() -> pd.DataFrame:
    """Génère les audiogrammes et retourne un DataFrame."""
    output_file = "data/tonal_exams.csv"

    if os.path.exists(output_file):
        os.remove(output_file)

    df = run_generation(10000, output_file)  # Nombre d'audiogrammes à générer

    return df  # On retourne le DataFrame pour Kedro


   

def replace_alphanumeric_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs alphanumériques (lettres) par une interpolation linéaire
    basée sur les valeurs before ou after.

    Args:
        data (pd.DataFrame): Jeu de données brut.

    Returns:
        pd.DataFrame: Jeu de données sans valeurs alphanumériques.
    """
    output_file = "data/intermediate_cleaned_datas.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    data_cleaned = data.copy()
    
    for col in data.columns:
        for i in range(1, len(data) - 1):
            curr_val = data[col].iloc[i]
            prev_val = data[col].iloc[i - 1]
            next_val = data[col].iloc[i + 1]
            
            # Vérifier si la valeur actuelle est une lettre
            if isinstance(curr_val, str) and not curr_val.isdigit():
                numeric_neighbors = []
                
                # Ajouter les valeurs voisines si elles sont numériques
                if isinstance(prev_val, (int, float)) and not pd.isna(prev_val):
                    numeric_neighbors.append(prev_val)
                if isinstance(next_val, (int, float)) and not pd.isna(next_val):
                    numeric_neighbors.append(next_val)
                
                # Si on a trouvé des voisins numériques, on remplace par leur moyenne
                if numeric_neighbors:
                    data_cleaned.at[i, col] = sum(numeric_neighbors) / len(numeric_neighbors)
                else:
                    data_cleaned.at[i, col] = np.nan  # Laisser NaN si aucun voisin n'est numérique
    
    # 🔥 Convertir toutes les colonnes en nombres après remplacement des valeurs
    data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')
    
    return data_cleaned


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les valeurs aberrantes et interpolant les valeurs manquantes.

    Args:
        data (pd.DataFrame): Jeu de données brut.

    Returns:
        pd.DataFrame: Jeu de données nettoyé.
    """

    output_file = "data/cleaned_data_final.csv"
    if os.path.exists(output_file):
        os.remove(output_file)

    # 🔥 Assurer que toutes les valeurs sont bien des nombres
    data = data.apply(pd.to_numeric, errors='coerce')

    # Suppression des valeurs aberrantes
    Q1 = data.quantile(0.15)
    Q3 = data.quantile(0.85)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Interpolation des valeurs manquantes
    data_cleaned = data_cleaned.interpolate(method="linear").fillna(0).round().astype(int)
    return data_cleaned