from kedro.pipeline import node, Pipeline, pipeline

from .nodes import compute_metrics, create_model, split_train_test, train_model  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=split_train_test,
                inputs="cleaned_data_final",
                outputs=["X_train", "X_test", "y_train", "y_test","shaped_data"],
                name="train_test_split",
            ),
        node(
                func=create_model,
                inputs="shaped_data",
                outputs="model",
                name="create_model",
            ),
        node(
                func=train_model,
                inputs=["model","X_train", "X_test", "y_train", "y_test"],
                outputs="trained_model",
                name="train_model",
            ),
        node(
                func=compute_metrics,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="metrics",
                name="compute_metrics",
            ),
    ])
