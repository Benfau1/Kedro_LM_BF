from kedro.pipeline import node
import pandas as pd

def generate_audiogram_data(data: pd.DataFrame):
    """Exécute le script de génération d'audiogrammes."""
    print(data)
    return data 



def clean_data(data: pd.DataFrame):
    """
    Nettoie les données en supprimant les valeurs aberrantes et interpolant les valeurs manquantes.

    Args:
        data (pd.DataFrame): Jeu de données brut.

    Returns:
        pd.DataFrame: Jeu de données nettoyé.
    

    # Suppression des valeurs aberrantes
    Q1 = data.quantile(0.15)
    Q3 = data.quantile(0.85)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Interpolation des valeurs manquantes
    data_cleaned = data_cleaned.interpolate(method="linear")

    return data_cleaned
    """