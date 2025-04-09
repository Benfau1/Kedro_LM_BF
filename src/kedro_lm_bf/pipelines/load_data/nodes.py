from kedro.pipeline import node
from kedro_lm_bf.data.audiogram_generator import run_generation
import pandas as pd

def generate_audiogram_data() -> pd.DataFrame:
    """Génère les audiogrammes et retourne un DataFrame."""
    output_file = "data/tonal_exams.csv"

    df = run_generation(10000, output_file)  # Nombre d'audiogrammes à générer

    return df  # On retourne le DataFrame pour Kedro