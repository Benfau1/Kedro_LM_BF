from kedro.pipeline import node, Pipeline, pipeline
from .nodes import generate_audiogram_data, clean_data, replace_alphanumeric_values


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=generate_audiogram_data, 
                inputs=None, 
                outputs="generated_csv", 
                name="generate_audiogram_data"),
        node(
                func=replace_alphanumeric_values, 
                inputs="generated_csv", 
                outputs="intermediate_cleaned_data", 
                name="replace_alphanumeric_values"),
        node(
                func=clean_data,
                inputs="intermediate_cleaned_data",
                outputs="cleaned_data_final",
                name="clean_data",
            ),
    ])
