import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score


def split_train_test(transformed_data):
     # Identifier les colonnes à prédire (celles commençant par 'after')
    target_columns = [col for col in transformed_data.columns if col.startswith("after")]
    feature_columns = [col for col in transformed_data.columns if col not in target_columns]

    # Séparer les features (X) et les labels (y)
    X = transformed_data[feature_columns]
    y = transformed_data[target_columns]

    # Séparer les données en train et test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Nombre de fréquences / 1
    before_columns_count = len([col for col in transformed_data.columns if col.startswith("before")])
    shaped_data = pd.DataFrame([], columns=[before_columns_count,1])

    return X_train, X_test, y_train, y_test, shaped_data

def create_model(input_shape, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):
    input_shape=(7,1)
    # Définition de la couche d'entrée
    inputs = layers.Input(shape=input_shape) # format (dim,1)
    # ML flow avant train

    # Définition des couches de convolution
    x = layers.Conv1D(filters=32, kernel_size=3, activation=activation)(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.ZeroPadding1D(padding=1)(x)  # Ajouter une couche de padding
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activation)(x)
    x = layers.ZeroPadding1D(padding=1)(x)  # Ajouter une couche de padding
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Aplatir les données
    x = layers.Flatten()(x)

    # Définition des couches entièrement connectées
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(input_shape[0], activation='softmax')(x)

    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="mse", metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model

def train_model(ml_model,X_train, X_test, y_train, y_test,epochs=10, batch_size=32,learning_rate=1e-3):
    # Entraîner le modèle
    ml_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    ml_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="mse", metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return ml_model

def compute_metrics(trained_model, X_test, y_test):
    # Prédictions sur les données de test
    y_pred = trained_model.predict(X_test)
    
    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Pour accuracy et F1-score, tu dois t'assurer que y_test et y_pred sont des classes (non des valeurs continues)
    # Si tu as un problème de classification, tu peux utiliser accuracy_score et f1_score
    if len(y_test.shape) == 2:  # Cas de classification
        y_test_classes = tf.argmax(y_test, axis=1).numpy()
        y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
    else:  # Cas de régression, on n'a pas d'accuracy ou de F1-score
        accuracy = None
        f1 = None
    
    # Stocker les résultats dans un DataFrame
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Accuracy": accuracy,
        "F1-score": f1
    }

    return pd.DataFrame(metrics, index=[0])