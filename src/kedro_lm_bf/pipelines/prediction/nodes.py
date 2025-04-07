import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

def predict(x_val: pd.DataFrame, y_val: pd.DataFrame, trained_model) -> pd.DataFrame:
    """
    Utilise le modèle entraîné pour prédire sur x_val, calcule le MAE moyen 
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
    predictions = trained_model.predict(x_val)
    
    # Calcul du MAE moyen sur toutes les sorties
    # Ici, on calcule l'erreur absolue sur chaque élément et on en fait la moyenne globale
    mae_mean = np.mean(np.abs(y_val.values - predictions))
    print(f"Mean Absolute Error moyen: {mae_mean:.4f}")
    
    # Calcul de l'accuracy (pourcentage de bonnes prédictions)
    # On convertit les sorties one-hot en étiquettes en prenant l'argmax sur l'axe des colonnes
    y_true_labels = np.argmax(y_val.values, axis=1)
    y_pred_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Construction d'un DataFrame résultat avec les vraies valeurs et les prédictions pour chaque sortie
    n_outputs = predictions.shape[1] if predictions.ndim == 2 else 1
    df_y_true = pd.DataFrame(y_val.values, columns=[f"y_true_{i}" for i in range(n_outputs)])
    df_y_pred = pd.DataFrame(predictions, columns=[f"y_pred_{i}" for i in range(n_outputs)])
    results_df = pd.concat([df_y_true, df_y_pred], axis=1)
    
    return results_df