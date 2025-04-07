import numpy as np
import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, mean_squared_error
import mlflow

mlflow.autolog()

def min_max_normalize(df):
    """
    Normalise chaque colonne avec la formule (x - min) / (max - min)
    """
    df_norm = df.copy()
    for col in df.columns:
        A = df[col].min()
        B = df[col].max()
        if B != A:
            df_norm[col] = (df[col] - A) / (B - A)
        else:
            df_norm[col] = 0.0  # éviter division par zéro
    return df_norm


def split_train_test(transformed_data):
    # Identifier les colonnes à prédire (celles commençant par 'after')
    target_columns = [col for col in transformed_data.columns if col.startswith("after")]
    feature_columns = [col for col in transformed_data.columns if col not in target_columns]

    # Séparer les features (X) et les labels (y)
    X = transformed_data[feature_columns]
    y = transformed_data[target_columns]

    # 🔹 Normalisation des features uniquement
    X = min_max_normalize(X)

    # Split Train / Test / Val
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(input_shape, 
                 output_shape, 
                 task_type='regression',  # 'classification' ou 'regression'
                 units=256, 
                 activation='relu', 
                 l2_value=1e-4, 
                 dropout_rate=0.3, 
                 learning_rate=1e-3):

    if isinstance(output_shape, pd.DataFrame):
        output_shape = output_shape.shape[1]

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
        loss = 'mae'
        metrics = ['mae']

    outputs = layers.Dense(output_shape, activation=output_activation)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss, metrics=metrics)

    return model

def train_model(ml_model, X_train, X_val, y_train, y_val, epochs=500, batch_size=32, learning_rate=1e-3):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    ml_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss="mae", metrics=["mae"])

    ml_model.fit(X_train, y_train,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stop, reduce_lr])
    
    return ml_model


def compute_metrics(trained_model, X_test, y_test):
    # Prédictions sur les données de test
    y_pred = trained_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse
    }

    return pd.DataFrame(metrics, index=[0])