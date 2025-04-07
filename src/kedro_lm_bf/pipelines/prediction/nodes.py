import pandas as pd
import numpy as np

def min_max_normalize_predict(x_val, X_min, X_max):
    """
    Normalise x_val en utilisant X_min et X_max fournis.
    """
    df_norm = x_val.copy()  # Copie de x_val pour éviter modification directe
    
    # S'assurer que X_min et X_max ont le même index que x_val
    X_min = pd.Series(X_min.values, index=x_val.columns)
    X_max = pd.Series(X_max.values, index=x_val.columns)
    
    for col in x_val.columns:
        A = X_min[col]  # Valeur min pour la normalisation
        B = X_max[col]  # Valeur max pour la normalisation
        if B != A:
            df_norm[col] = (x_val[col] - A) / (B - A)  # Normalisation
        else:
            df_norm[col] = 0.0  # Pour éviter la division par zéro
    
    return df_norm

def predict(x_val: pd.DataFrame, trained_model, X_min, X_max) -> pd.DataFrame:
    """
    Fonction de prédiction avec normalisation des features de x_val en utilisant X_min et X_max.
    """
    # Normalisation de x_val
    df_norm = min_max_normalize_predict(x_val, X_min, X_max)
    
    # Reformater les données pour Conv1D (batch_size, timesteps, features)
    x_reshaped = df_norm.values.reshape(df_norm.shape[0], df_norm.shape[1], 1)
    
    # Prédiction avec le modèle
    predictions_norm = trained_model.predict(x_reshaped)
    
    # Dé-normalisation des prédictions
    # Utiliser les mêmes X_min et X_max que pour la normalisation (car le modèle prédit les mêmes features)
    y_range = (X_max.values - X_min.values).reshape(1, -1)
    y_min_values = X_min.values.reshape(1, -1)
    
    # S'assurer que les dimensions sont compatibles
    if predictions_norm.shape[1] != y_range.shape[1]:
        # Adaptation en fonction du nombre de sorties du modèle
        predictions = predictions_norm  # Sans dénormalisation si les dimensions ne correspondent pas
    else:
        predictions = predictions_norm * y_range + y_min_values
    
    # Construction d'un DataFrame avec les prédictions
    if predictions.shape[1] == 1:
        results_df = pd.DataFrame(predictions, columns=["y_pred"])
    else:
        results_df = pd.DataFrame(predictions, columns=[f"y_pred_{i}" for i in range(predictions.shape[1])])
    
    return results_df