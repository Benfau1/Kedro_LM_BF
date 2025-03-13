from kedro.pipeline import node, Pipeline, pipeline

from kedro_lm_bf.pipelines.load_data.nodes import clean_data  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=clean_data,
                inputs="raw_data",
                outputs="cleaned_data",
                name="data_cleaning_node",
            ),
    ])
