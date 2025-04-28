import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def predict(data_to_predict: pd.DataFrame, trained_model):
    # Ne garder que les colonnes "before"
    before_columns = [col for col in data_to_predict.columns if "before" in col]
    x_to_predict = data_to_predict[before_columns]

    # Adapter la forme pour le modèle
    x_to_predict = np.expand_dims(x_to_predict.values, axis=-1)

    # Faire la prédiction
    predictions = trained_model.predict(x_to_predict)

    # Créer un DataFrame pour les prédictions
    predictions_df = pd.DataFrame(predictions, 
                                  columns=[
                                      "after_exam_125_Hz",
                                      "after_exam_250_Hz",
                                      "after_exam_500_Hz",
                                      "after_exam_1000_Hz",
                                      "after_exam_2000_Hz",
                                      "after_exam_4000_Hz",
                                      "after_exam_8000_Hz"])

    # Retourner les données originales + les prédictions
    return pd.concat([data_to_predict, predictions_df], axis=1)
