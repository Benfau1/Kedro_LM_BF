import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def predict(x_val: pd.DataFrame, y_val: pd.DataFrame, trained_model, y_min, y_max) -> pd.DataFrame:
    """
    Utilise le modèle entraîné pour prédire sur x_val, calcule le MAE moyen et le score R² moyen
    sur chaque sortie et le pourcentage d'accuracy (comparaison des étiquettes),
    puis retourne un DataFrame combinant les valeurs réelles et prédites.

    Args:
        x_val (pd.DataFrame): Données d'entrée pour la prédiction.
        y_val (pd.DataFrame): Vraies valeurs (one-hot encoded).
        trained_model: Modèle entraîné.

    Returns:
        pd.DataFrame: DataFrame combinant "y_true" et "y_pred" pour chaque sortie.
    """
    # Prédire avec le modèle
    predictions_norm = trained_model.predict(x_val)
    # Convertir en array avec forme (1, -1) pour broadcast
    y_range = (y_max.values - y_min.values).reshape(1, -1)
    y_min_values = y_min.values.reshape(1, -1)

    predictions = predictions_norm * y_range + y_min_values
    
    # Calcul du MAE moyen sur toutes les sorties
    # Ici, on calcule l'erreur absolue sur chaque élément et on en fait la moyenne globale
    mae_mean = np.mean(np.abs(y_val.values - predictions))
    print(f"Mean Absolute Error moyen: {mae_mean:.4f}")

    r2_mean = r2_score(y_val.values, predictions)
    print(f"R² Score moyen: {r2_mean:.4f}")
    
    # Construction d'un DataFrame résultat avec les vraies valeurs et les prédictions pour chaque sortie
    n_outputs = predictions.shape[1] if predictions.ndim == 2 else 1
    df_y_true = pd.DataFrame(y_val.values, columns=[f"y_true_{i}" for i in range(n_outputs)])
    df_y_pred = pd.DataFrame(predictions, columns=[f"y_pred_{i}" for i in range(n_outputs)])
    results_df = pd.concat([df_y_true, df_y_pred], axis=1)
    
    return results_df