from kedro.pipeline import node, Pipeline, pipeline
from .nodes import predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict,
            inputs=["data_to_predict", "trained_model"],
            outputs="predictions",
            name="predict_node",
        )
    ])