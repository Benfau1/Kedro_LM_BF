from kedro.pipeline import node, Pipeline, pipeline

from .nodes import compute_metrics, create_model, min_max_normalize, split_train_test, train_model  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=split_train_test,
                inputs="cleaned_data_final",
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "y_min", "y_max"],
                name="train_test_split",
            ),
        node(
                func=create_model,
                inputs=dict(input_shape="X_train", output_shape="y_train"),
                outputs="model",
                name="create_model",
            ),
        node(
                func=train_model,
                inputs=["model","X_train", "X_val", "y_train", "y_val"],
                outputs="trained_model",
                name="train_model",
            ),
        node(
                func=compute_metrics,
                inputs=["trained_model", "X_test", "y_test", "y_min", "y_max"],
                outputs="metrics",
                name="compute_metrics",
            ),
    ])
