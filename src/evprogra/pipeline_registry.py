"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

from evprogra.pipelines.data_ingestion import create_pipeline as create_data_ingestion
from evprogra.pipelines.data_cleaning import create_pipeline as create_data_cleaning
from evprogra.pipelines.data_transform import create_pipeline as create_data_transform
from evprogra.pipelines.data_validation import create_pipeline as create_data_validation


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    data_ingestion = create_data_ingestion()
    data_cleaning = create_data_cleaning()
    data_transform = create_data_transform()
    data_validation = create_data_validation()

    return {
        "data_ingestion": data_ingestion,
        "data_cleaning": data_cleaning,
        "data_transform": data_transform,
        "data_validation": data_validation,
        "__default__": data_cleaning + data_transform + data_validation,
    }