"""
Pipeline: data_ingestion — registro de nodos
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import cargar_y_explorar_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=cargar_y_explorar_datasets,
                inputs=[
                    "empleados",
                    "ausencias",
                    "capacitaciones",
                    "evaluaciones",
                ],
                outputs="reporte_ingestion",
                name="cargar_y_explorar_datasets_node",
            ),
        ]
    )
