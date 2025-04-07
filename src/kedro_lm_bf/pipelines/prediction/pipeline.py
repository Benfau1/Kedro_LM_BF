from kedro.pipeline import node, Pipeline, pipeline
from .nodes import predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict,
            inputs=["x_val", "y_val", "trained_model"],
            outputs="predictions",
            name="predict_node",
        ),
    ])