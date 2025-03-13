from kedro.pipeline import node, Pipeline, pipeline

from .nodes import create_model, train_test_split  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=train_test_split,
                inputs="transformed_data",
                outputs=["train_data", "test_data"],
                name="train_test_split",
            ),
        node(
                func=create_model,
                inputs="train_data",
                outputs="cleaned_data",
                name="create_model",
            ),
    ])
