from kedro.pipeline import node, Pipeline, pipeline
from kedro_lm_bf.pipelines.load_data.nodes import clean_data  # noqa
from .nodes import generate_audiogram_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=generate_audiogram_data, 
                inputs=None, 
                outputs="generated_csv", 
                name="generate_audiograms"),
        node(
                func=clean_data,
                inputs="generated_csv",
                outputs="cleaned_data",
                name="data_cleaning_node",
            ),
    ])
