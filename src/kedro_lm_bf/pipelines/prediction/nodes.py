import pandas as pd
import numpy as np

def predict(data_to_predict: pd.DataFrame, X_min: pd.Series, X_max: pd.Series, trained_model):

    # Séparer les colonnes "before" et "after"
    before_columns = [col for col in data_to_predict.columns if "before" in col]

    # On garde uniquement les colonnes "before" pour la prédiction
    X_to_predict = data_to_predict[before_columns].copy()

    # Normaliser X à prédire
    X_norm = (X_to_predict.values - X_min.values.reshape(1, -1)) / (X_max.values.reshape(1, -1) - X_min.values.reshape(1, -1))
    X_norm = X_norm.reshape(X_norm.shape[0], X_norm.shape[1], 1)

    # Faire la prédiction avec le modèle
    predictions = trained_model.predict(X_norm)

    # Dénormaliser les prédictions
    predictions_denorm = predictions * (X_max.values.reshape(1, -1) - X_min.values.reshape(1, -1)) + X_min.values.reshape(1, -1)

    # Retourner les prédictions dénormalisées sous forme de DataFrame
    return pd.DataFrame(predictions_denorm)