"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro_lm_bf.pipelines.load_data import pipeline as load_data_pipeline
from kedro_lm_bf.pipelines.train_data import pipeline as train_data_pipeline
from kedro_lm_bf.pipelines.prediction import pipeline as prediction_pipeline
from kedro_lm_bf.pipelines.transform_data import pipeline as transform_data_pipeline
from kedro_lm_bf.pipelines.deploy import pipeline as deploy_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    pipelines["load_data"] = load_data_pipeline.create_pipeline()
    pipelines["transform_data"] = transform_data_pipeline.create_pipeline()
    pipelines["entrainement"] = train_data_pipeline.create_pipeline()
    pipelines["prediction"] = prediction_pipeline.create_pipeline()
    pipelines["deployment"] = deploy_pipeline.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())

    return pipelines