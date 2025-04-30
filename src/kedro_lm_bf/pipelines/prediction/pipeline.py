from kedro.pipeline import Pipeline, node, pipeline
from kedro_lm_bf.pipelines.prediction.nodes import predict

# Fonction qui crée la pipeline de prédiction
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict,  # Fonction à exécuter dans ce nœud
            inputs=["data_to_predict", "X_min", "X_max", "trained_model"],  # Données d’entrée du nœud
            outputs="predictions",  # Nom du jeu de données en sortie
            name="predict_node",  # Nom du nœud (utile pour les logs et les filtres)
        )
    ])