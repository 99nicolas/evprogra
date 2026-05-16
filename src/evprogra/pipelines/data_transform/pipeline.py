"""
pipeline.py — Pipeline data_transform
Caso 3: Recursos Humanos — SCY1101
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    integrar_datasets,
    calcular_resumen_departamento,
    transformar_y_codificar,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=integrar_datasets,
                inputs=[
                    "empleados_clean",
                    "ausencias_clean",
                    "capacitaciones_clean",
                    "evaluaciones_clean",
                    "params:transform",
                ],
                outputs="rrhh_integrado",
                name="integrar_datasets_node",
            ),
            node(
                func=calcular_resumen_departamento,
                inputs="rrhh_integrado",
                outputs="resumen_departamento",
                name="calcular_resumen_departamento_node",
            ),
            node(
                func=transformar_y_codificar,
                inputs=["rrhh_integrado", "params:transform"],
                outputs="rrhh_encoded",
                name="transformar_y_codificar_node",
            ),
        ]
    )