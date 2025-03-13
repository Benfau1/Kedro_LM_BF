from kedro.pipeline import node, Pipeline, pipeline

from kedro_lm_bf.pipelines import train_data  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=train_data,
                inputs="train_data",
                outputs="cleaned_data",
                name="data_cleaning_node",
            ),
    ])
