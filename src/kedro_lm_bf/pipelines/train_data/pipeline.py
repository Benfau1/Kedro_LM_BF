from kedro.pipeline import node, Pipeline, pipeline

from kedro_lm_bf.pipelines import train_data  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
    ])
