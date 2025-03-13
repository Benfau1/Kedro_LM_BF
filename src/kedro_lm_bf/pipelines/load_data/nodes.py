import pandas as pd
from data.audiogram_generator import run_generation
def generate_audiogram_data():
    """Exécute le script de génération d'audiogrammes."""
    csv_filename = "data/tonal_exams.csv"  
    exam_count = 100000  
    run_generation(exam_count, csv_filename)
    return csv_filename 



def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les valeurs aberrantes et interpolant les valeurs manquantes.

    Args:
        data (pd.DataFrame): Jeu de données brut.

    Returns:
        pd.DataFrame: Jeu de données nettoyé.
    """

    # Suppression des valeurs aberrantes
    Q1 = data.quantile(0.15)
    Q3 = data.quantile(0.85)
    IQR = Q3 - Q1
    data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Interpolation des valeurs manquantes
    data_cleaned = data_cleaned.interpolate(method="linear")

    return data_cleaned
