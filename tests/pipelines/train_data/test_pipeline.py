import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from kedro_lm_bf.pipelines.train_data import nodes


# Vérifie que la normalisation min-max fonctionne correctement
def test_min_max_normalize():
    df = pd.DataFrame({
        "before_1": [10, 20, 30],
        "before_2": [0, 5, 10],
        "before_3": [5, 5, 5],   # constante
        "before_4": [100, 50, 0],
        "before_5": [1, 2, 3],
        "before_6": [7, 8, 9],
        "before_7": [10, 10, 10],  # constante
    })
    norm_df, min_vals, max_vals = nodes.min_max_normalize(df)

    assert (norm_df["before_1"].min() == 0) and (norm_df["before_1"].max() == 1)
    assert (norm_df["before_3"] == 0.0).all()
    assert (norm_df["before_7"] == 0.0).all()
    print("test_min_max_normalize passé ")

# Vérifie que les dimensions des splits train/val/test sont cohérentes
def test_split_train_test_shapes():
    df = pd.DataFrame({
        **{f"before_{i+1}": np.random.rand(100) * 120 for i in range(7)},
        **{f"after_{i+1}": np.random.rand(100) * 120 for i in range(7)}
    })
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = nodes.split_train_test(df)

    assert X_train.shape[1] == 7
    assert y_train.shape[1] == 7
    assert len(X_train) > len(X_val) > 0 and len(X_test) > 0
    print(f"test_split_train_test_shapes : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# Vérifie que le modèle renvoie une sortie avec le bon nombre de neurones
def test_create_model_output_shape():
    model = nodes.create_model(input_shape=(7, 1), output_shape=7)
    assert model.output_shape[-1] == 7
    print(f"test_create_model_output_shape : output shape = {model.output_shape}")

# Vérifie que le modèle peut être entraîné sans erreur
def test_train_model_executes():
    X = np.random.rand(20, 7, 1)
    y = np.random.rand(20, 7)
    model = nodes.create_model((7, 1), 7)
    trained_model = nodes.train_model(model, X, X, y, y, epochs=2, batch_size=5)

    assert hasattr(trained_model, "predict")
    print("test_train_model_executes : entraînement terminé")

# Vérifie que le calcul des métriques retourne bien les colonnes attendues
def test_compute_metrics_format():
    X = np.random.rand(10, 7, 1)
    y = np.random.rand(10, 7)
    model = nodes.create_model((7, 1), 7)
    model.fit(X, y, epochs=2, verbose=0)

    y_min = pd.Series([0]*7)
    y_max = pd.Series([1]*7)
    df = nodes.compute_metrics(model, X, pd.DataFrame(y), y_min, y_max)

    assert "MAE" in df.columns and "R2" in df.columns
    assert df.shape == (1, 2)
    print(f"test_compute_metrics_format : MAE = {df['MAE'].iloc[0]:.4f}, R2 = {df['R2'].iloc[0]:.4f}")

# Vérifie que les callbacks early stopping et reduce_lr sont bien utilisés
def test_callbacks_effect():
    X = np.random.rand(60, 7, 1)
    y = np.random.rand(60, 7)
    model = nodes.create_model((7, 1), 7)

    history = model.fit(
        X, y,
        validation_data=(X, y),
        epochs=50,
        batch_size=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
        ],
        verbose=0,
    )

    assert len(history.history['loss']) < 50, "Early stopping non déclenché"
    print(f"test_callbacks_effect : entraînement stoppé à epoch {len(history.history['loss'])}")
    K.clear_session()