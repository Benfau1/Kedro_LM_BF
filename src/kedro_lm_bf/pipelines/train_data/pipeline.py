from kedro.pipeline import node, Pipeline, pipeline

from .nodes import compute_metrics, create_model, min_max_normalize, split_train_test, train_model  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=min_max_normalize,
                inputs="cleaned_data_final",
                outputs="test",
                name="min_max_normalize",
            ),
        node(
                func=split_train_test,
                inputs="test",
                outputs=["X_train", "x_val", "X_test", "y_train", "y_val", "y_test", "shaped_data"],
                name="train_test_split",
            ),
        node(
                func=create_model,
                inputs=["shaped_data","y_train"],
                outputs="model",
                name="create_model",
            ),
        node(
                func=train_model,
                inputs=["model","X_train", "x_val", "y_train", "y_val"],
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
