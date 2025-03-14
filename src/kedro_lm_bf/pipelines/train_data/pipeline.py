from kedro.pipeline import node, Pipeline, pipeline

from .nodes import create_model, split_train_test  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=split_train_test,
                inputs="transformed_data",
                outputs=["train_data", "test_data","shaped_data"],
                name="train_test_split",
            ),
        node(
                func=create_model,
                inputs="shaped_data",
                outputs="ml_model",
                name="create_model",
            ),
    ])
