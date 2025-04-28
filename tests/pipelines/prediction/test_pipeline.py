import numpy as np
import pandas as pd
from kedro_lm_bf.pipelines.prediction import nodes

def test_predict_output_shape():
    # Données simulées avec 7 colonnes "before" et 7 colonnes "after"
    X_val = pd.DataFrame(
        np.random.rand(10, 14),
        columns=[f"before_{i}" if i < 7 else f"after_{i-7}" for i in range(14)]
    )

    # Simule un modèle entraîné
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input

    model = Sequential([
        Input(shape=(7, 1)),
        Flatten(),
        Dense(7)
    ])
    model.compile(optimizer='adam', loss='mse')

    predictions = nodes.predict(X_val, model)

    assert isinstance(predictions, pd.DataFrame), "La prédiction doit renvoyer un DataFrame"
    assert predictions.shape[0] == X_val.shape[0], "Nombre de lignes des prédictions incorrect"
    print("Test de prédiction passé avec succès")