import numpy as np
import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error
import mlflow

mlflow.autolog()

def split_train_test(transformed_data):
    # Identifier les colonnes à prédire (celles commençant par 'after')
    target_columns = [col for col in transformed_data.columns if col.startswith("after")]
    feature_columns = [col for col in transformed_data.columns if col not in target_columns]

    # Séparer les features (X) et les labels (y)
    X = transformed_data[feature_columns]
    y = transformed_data[target_columns]

    # Première séparation: Train (80%) / Test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Deuxième séparation: Validation (5% du total) => 5/80 = 6.25% du train temporaire
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

    # Info sur les dimensions d’entrée
    before_columns_count = len([col for col in transformed_data.columns if col.startswith("before")])
    shaped_data = pd.DataFrame([], columns=[before_columns_count, 1])

    return X_train, X_val, X_test, y_train, y_val, y_test, shaped_data


def create_model(input_shape, 
                 output_shape, 
                 task_type='regression',  # 'classification' ou 'regression'
                 units=256, 
                 activation='relu', 
                 l2_value=1e-4, 
                 dropout_rate=0.3, 
                 learning_rate=1e-3):

    inputs = layers.Input(shape=(7, 1))

    # Bloc convolution
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation=activation)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation=activation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Global pooling au lieu de Flatten
    x = layers.GlobalAveragePooling1D()(x)

    # Dense + régularisation + dropout
    x = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(l2_value))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Sortie
    if task_type == 'classification':
        output_activation = 'softmax'
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:  # regression
        output_activation = 'linear'
        loss = 'mse'
        metrics = ['mae']

    outputs = layers.Dense(output_shape.shape[1], activation=output_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss, metrics=metrics)

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
    
    # Pour accuracy et F1-score, tu dois t'assurer que y_test et y_pred sont des classes (non des valeurs continues)
    # Si tu as un problème de classification, tu peux utiliser accuracy_score et f1_score
    if len(y_test.shape) == 2:  # Cas de classification
        y_test_classes = tf.argmax(y_test, axis=1).numpy()
        y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    else:  # Cas de régression, on n'a pas d'accuracy ou de F1-score
        accuracy = None
        f1 = None
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        "Accuracy": accuracy,
        "F1-score": f1,
        "MAE": mae,
        "RMSE": rmse
    }

    return pd.DataFrame(metrics, index=[0])