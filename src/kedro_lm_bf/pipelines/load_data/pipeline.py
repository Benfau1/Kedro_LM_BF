from kedro.pipeline import node, Pipeline, pipeline
from .nodes import generate_audiogram_data, clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=generate_audiogram_data, 
                inputs="raw_data", 
                outputs="generated_csv", 
                name="generate_audiogram_data"),
        #node(
        #        func=clean_data,
        #        inputs="generated_csv",
        #        outputs="cleaned_data",
        #        name="clean_data",
        #    ),
    ])
