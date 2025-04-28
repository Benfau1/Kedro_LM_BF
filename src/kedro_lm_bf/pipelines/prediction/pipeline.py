from kedro.pipeline import Pipeline, node, pipeline
from kedro_lm_bf.pipelines.prediction.nodes import predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict,
            inputs=["data_to_predict", "X_min", "X_max", "trained_model"],
            outputs="predictions",
            name="predict_node",
        )
    ])