"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=predict,
                inputs="data_to_predict",
                name="predict",
            ),
    ])
