"""
This is a boilerplate pipeline 'prediction'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline

from kedro_lm_bf.pipelines.prediction.nodes import predict  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
    ])
