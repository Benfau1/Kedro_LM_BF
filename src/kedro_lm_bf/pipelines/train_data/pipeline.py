from kedro.pipeline import node, Pipeline, pipeline

# Importation des fonctions définies dans nodes.py, qui réalisent les différentes étapes du pipeline
from .nodes import compute_metrics, create_model, min_max_normalize, split_train_test, train_model  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    # Création et retour d'un pipeline Kedro composé d'une séquence de noeuds
    return pipeline([
        # Premier noeud : division des données en jeu d'entraînement, validation et test
        node(
                func=split_train_test,
                inputs="data_to_predict",  # données brutes à diviser
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "y_min", "y_max","X_min", "X_max"],  # jeux de données et min/max pour dénormalisation
                name="train_test_split",  # nom du noeud pour le suivi
            ),
        # Deuxième noeud : création de la structure du modèle
        node(
                func=create_model,
                inputs=dict(input_shape="X_train", output_shape="y_train"),  # les formes des données sont utilisées pour définir le modèle
                outputs="model",  # sortie du modèle non encore entraîné
                name="create_model",
            ),
        # Troisième noeud : entraînement du modèle avec les données d'entraînement et de validation
        node(
                func=train_model,
                inputs=["model","X_train", "X_val", "y_train", "y_val"],  # données nécessaires à l'entraînement
                outputs="trained_model",  # modèle entraîné prêt à être utilisé
                name="train_model",
            ),
        # Quatrième noeud : évaluation du modèle avec les données de test
        node(
                func=compute_metrics,
                inputs=["trained_model", "X_test", "y_test", "y_min", "y_max"],  # modèle et données de test + infos pour dénormalisation
                outputs="metrics",  # résultats des métriques (ex: MAE, R²)
                name="compute_metrics",
            ),
    ])
