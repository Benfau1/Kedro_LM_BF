from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, replace_alphanumeric_values

# This function defines the transformation pipeline that cleans and prepares data.
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Replace alphanumeric characters with clean numeric values
        node(
                func=replace_alphanumeric_values, 
                inputs="generated_csv", 
                outputs="intermediate_cleaned_data", 
                name="replace_alphanumeric_values"
            ),
        # Further clean the data and prepare it for prediction
        node(
                func=clean_data,
                inputs="intermediate_cleaned_data",
                outputs="data_to_predict",
                name="clean_data",
            )
    ])
