from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, replace_alphanumeric_values

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=replace_alphanumeric_values, 
                inputs="generated_csv", 
                outputs="intermediate_cleaned_data", 
                name="replace_alphanumeric_values"
            ),
        node(
                func=clean_data,
                inputs="intermediate_cleaned_data",
                outputs="data_to_predict",
                name="clean_data",
            )
    ])
