import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def predict(x_val: pd.DataFrame, y_val, trained_model, X_min, X_max, y_min, y_max) :
    # Prédiction
    predictions_norm = trained_model.predict(x_val)

    y_true_array = y_val.values.ravel()
    y_pred_array = predictions_norm.ravel()

    r2 = r2_score(y_true_array, y_pred_array)

    print(f"R² score: {r2:.4f}")
    return
