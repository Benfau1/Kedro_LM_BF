import tensorflow as tf
from keras import layers, regularizers
import pandas as pd

def train_test_split(transformed_data):
    # Nettoyer les données : conversion en numérique et suppression des valeurs non valides
    transformed_data = transformed_data.apply(pd.to_numeric, errors='coerce')
    transformed_data = transformed_data.dropna()

    # Séparer les données en train et test (80% / 20%)
    train_data, test_data = train_test_split(transformed_data, test_size=0.2, random_state=42)
    return train_data, test_data

def create_model(train_data, units=128, activation='relu', l2_value=0.01, dropout_rate=None, learning_rate=1e-3):

    # Définition de la couche d'entrée
    inputs = layers.Input(shape=train_data) # format (dim,1)

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

    x = layers.Dense(train_data[0], activation='softmax')(x)

    # Création du modèle
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss="mse", metrics=[tf.keras.metrics.CategoricalAccuracy()])
    
    return model
