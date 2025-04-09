from kedro.pipeline import node, Pipeline, pipeline
from .nodes import generate_audiogram_data, clean_data, replace_alphanumeric_values


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=generate_audiogram_data, 
                inputs=None, 
                outputs="generated_csv", 
                name="generate_audiogram_data")
    ])
