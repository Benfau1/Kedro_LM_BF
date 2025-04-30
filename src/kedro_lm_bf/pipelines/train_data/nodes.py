import numpy as np
import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow

# Activation de l'autologging de MLflow pour suivre automatiquement les paramètres, métriques et modèles
mlflow.autolog()

# Cette fonction effectue une normalisation min-max colonne par colonne
def min_max_normalize(df):
    """
    Normalise chaque colonne avec la formule (x - min) / (max - min)
    """
    df_norm = df.copy()
    min_vals = df.min()
    max_vals = df.max()
    # Pour chaque colonne, on applique la normalisation min-max
    for col in df.columns:
        A = min_vals[col]
        B = max_vals[col]
        if B != A:
            df_norm[col] = (df[col] - A) / (B - A)
        else:
            df_norm[col] = 0.0  # éviter division par zéro quand max = min
    return df_norm, min_vals, max_vals


# Cette fonction sépare les données transformées en ensembles d'entraînement, validation et test
def split_train_test(transformed_data):
    # Identifier les colonnes cibles (celles commençant par 'after')
    target_columns = [col for col in transformed_data.columns if col.startswith("after")]
    # Identifier les colonnes de caractéristiques (features)
    feature_columns = [col for col in transformed_data.columns if col not in target_columns]

    # Séparer les features (X) et les labels (y)
    X = transformed_data[feature_columns]
    y = transformed_data[target_columns]

    # Normalisation des features uniquement
    X, X_min, X_max = min_max_normalize(X)
    # Normalisation des labels
    y, y_min, y_max = min_max_normalize(y)

    # Split des données en train+val et test (80% train+val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split des données train+val en train et val (85% train, 15% val)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

    # Retourne les jeux de données ainsi que les min et max pour la normalisation inverse
    return X_train, X_val, X_test, y_train, y_val, y_test, y_min, y_max, X_min, X_max


# Cette fonction crée un modèle TensorFlow Keras pour une tâche de régression ou classification
def create_model(input_shape, #Paramètre optimisé grâce à mlflow
                 output_shape, 
                 task_type='regression',  #'regression'
                 units=512, 
                 activation='relu', 
                 l2_value=1e-4, 
                 dropout_rate=0.2, 
                 learning_rate=1e-4):

    # Si output_shape est un DataFrame, on récupère le nombre de colonnes comme taille de sortie
    if isinstance(output_shape, pd.DataFrame):
        output_shape = output_shape.shape[1]

    # Définition de l'entrée du modèle avec une forme fixe (7, 1)
    inputs = layers.Input(shape=(7, 1))

    # Bloc convolutionnel 1D avec 64 filtres, activation et normalisation batch
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation=activation)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)  # Réduction de la dimension temporelle

    # Deuxième bloc convolutionnel avec 128 filtres
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation=activation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Global average pooling pour réduire la dimensionnalité sans flatten
    x = layers.GlobalAveragePooling1D()(x)

    # Couche dense avec régularisation L2 et dropout pour éviter le surapprentissage
    x = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2_value))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Configuration de la couche de sortie selon le type de tâche
    if task_type == 'classification':
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:  # régression
        output_activation = 'linear'
        loss = 'mae'
        metrics = ['mae']

    # Couche de sortie avec la taille correspondante au nombre de variables cibles
    outputs = layers.Dense(output_shape, activation=output_activation)(x)

    # Création et compilation du modèle Keras
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss, metrics=metrics)

    return model


# Cette fonction entraîne le modèle avec les données d'entraînement et de validation
def train_model(ml_model, X_train, X_val, y_train, y_val, epochs=90, batch_size=32, learning_rate=1e-3):
    # Callback pour arrêter l'entraînement si la validation ne s'améliore plus
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # Callback pour réduire le taux d'apprentissage si la validation stagne
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Compilation du modèle avec l'optimiseur Adam et la fonction de perte MAE
    ml_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss="mae", metrics=["mae"])

    # Entraînement du modèle avec validation et callbacks
    ml_model.fit(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stop, reduce_lr])
    
    return ml_model


# Cette fonction calcule les métriques d'évaluation sur le jeu de test et affiche les résultats
def compute_metrics(trained_model, X_test, y_test, y_min, y_max):
    # Prédiction sur le jeu de test
    y_pred = trained_model.predict(X_test)

    # Calcul de la plage des valeurs pour la dénormalisation
    y_range = (y_max.values - y_min.values).reshape(1, -1)
    y_min_values = y_min.values.reshape(1, -1)

    # Dénormalisation des prédictions et des valeurs réelles
    y_pred_denorm = y_pred * y_range + y_min_values
    y_test_denorm = y_test.values * y_range + y_min_values

    # Calcul des métriques MAE et R2
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    r2 = r2_score(y_test_denorm, y_pred_denorm)

    # Affichage des résultats
    print(f"\n Résultats des métriques :")
    print(f"   ➤ MAE (Mean Absolute Error) : {mae:.4f}")
    print(f"   ➤ R² (Score de détermination) : {r2:.4f}")

    # Enregistrement des métriques dans MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # Retourne un DataFrame contenant les métriques
    return pd.DataFrame({
        "MAE": [mae],
        "R2": [r2]
    })
