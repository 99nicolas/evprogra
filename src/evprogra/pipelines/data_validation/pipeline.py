"""
pipeline.py — Pipeline: data_validation
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generar_reporte_validacion, generar_reporte_integridad


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generar_reporte_validacion,
            inputs=[
                "empleados", "ausencias", "capacitaciones", "evaluaciones",
                "empleados_clean", "ausencias_clean",
                "capacitaciones_clean", "evaluaciones_clean",
            ],
            outputs="reporte_validacion",
            name="generar_reporte_validacion_node",
        ),
        node(
            func=generar_reporte_integridad,
            inputs=[
                "empleados_clean", "ausencias_clean",
                "capacitaciones_clean", "evaluaciones_clean",
                "rrhh_encoded",
            ],
            outputs="reporte_integridad",
            name="generar_reporte_integridad_node",
        ),
    ])