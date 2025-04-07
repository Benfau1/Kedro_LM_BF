from kedro.pipeline import node, Pipeline, pipeline
from .nodes import predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict,
            inputs=["x_val", "y_val", "trained_model", "X_min", "X_max", "y_min", "y_max"],
            outputs="predictions",
            name="predict_node",
        ),
    ])