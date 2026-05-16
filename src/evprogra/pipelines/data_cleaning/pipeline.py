"""
pipeline.py — Pipeline: data_cleaning
=======================================
Define el grafo de nodos para la etapa de limpieza de datos.
Corresponde a AD 1.2 del curso SCY1101.

Flujo:
  [01_raw] empleados, ausencias, capacitaciones, evaluaciones
      ↓  (params:cleaning)
  [02_intermediate] *_clean
  [08_reporting] reporte_diagnostico
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    limpiar_empleados,
    limpiar_ausencias,
    limpiar_capacitaciones,
    limpiar_evaluaciones,
    generar_reporte_diagnostico,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ------------------------------------------------------------------
            # Nodo 0: Diagnóstico inicial ANTES de limpiar
            # ------------------------------------------------------------------
            node(
                func=generar_reporte_diagnostico,
                inputs=[
                    "empleados",
                    "ausencias",
                    "capacitaciones",
                    "evaluaciones",
                ],
                outputs="reporte_diagnostico",
                name="generar_reporte_diagnostico_node",
                tags=["diagnostico", "cleaning"],
            ),

            # ------------------------------------------------------------------
            # Nodo 1: Limpiar empleados
            # ------------------------------------------------------------------
            node(
                func=limpiar_empleados,
                inputs=["empleados", "params:cleaning"],
                outputs="empleados_clean",
                name="limpiar_empleados_node",
                tags=["cleaning"],
            ),

            # ------------------------------------------------------------------
            # Nodo 2: Limpiar ausencias
            # ------------------------------------------------------------------
            node(
                func=limpiar_ausencias,
                inputs=["ausencias", "params:cleaning"],
                outputs="ausencias_clean",
                name="limpiar_ausencias_node",
                tags=["cleaning"],
            ),

            # ------------------------------------------------------------------
            # Nodo 3: Limpiar capacitaciones
            # ------------------------------------------------------------------
            node(
                func=limpiar_capacitaciones,
                inputs=["capacitaciones", "params:cleaning"],
                outputs="capacitaciones_clean",
                name="limpiar_capacitaciones_node",
                tags=["cleaning"],
            ),

            # ------------------------------------------------------------------
            # Nodo 4: Limpiar evaluaciones
            # ------------------------------------------------------------------
            node(
                func=limpiar_evaluaciones,
                inputs=["evaluaciones", "params:cleaning"],
                outputs="evaluaciones_clean",
                name="limpiar_evaluaciones_node",
                tags=["cleaning"],
            ),
        ]
    )
