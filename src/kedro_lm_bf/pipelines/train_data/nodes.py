import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(transformed_data):
    # Nettoyer les données : conversion en numérique et suppression des valeurs non valides
    transformed_data = transformed_data.apply(pd.to_numeric, errors='coerce')
    transformed_data = transformed_data.dropna()

    # Séparer les données en train et test (80% / 20%)
    train_data, test_data = train_test_split(transformed_data, test_size=0.2, random_state=42)

    # Nombre de fréquences / 1
    before_columns_count = len([col for col in transformed_data.columns if col.startswith("before")])
    shaped_data = pd.DataFrame([], columns=[before_columns_count,1])

    return train_data, test_data, shaped_data

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
