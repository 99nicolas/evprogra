"""
pipeline.py — data_preprocessing
=================================
Registra los nodos de split en el grafo de Kedro.
Lee parámetros desde conf/base/parameters.yml (sección: preprocessing).
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preparar_datos_clasificacion, preparar_datos_regresion


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de preprocesamiento y split de datos."""
    return pipeline(
        [
            # ── Nodo 1: Split clasificación ─────────────────────────────
            node(
                func=preparar_datos_clasificacion,
                inputs=[
                    "rrhh_encoded",
                    "params:preprocessing.target_clf",
                    "params:preprocessing.drop_columns",
                    "params:preprocessing.test_size",
                    "params:preprocessing.random_state",
                    "params:preprocessing.stratify_clf",
                ],
                outputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                name="node_split_clasificacion",
            ),
            # ── Nodo 2: Split regresión ──────────────────────────────────
            node(
                func=preparar_datos_regresion,
                inputs=[
                    "rrhh_encoded",
                    "params:preprocessing.target_reg",
                    "params:preprocessing.drop_columns",
                    "params:preprocessing.test_size",
                    "params:preprocessing.random_state",
                ],
                outputs=["X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg"],
                name="node_split_regresion",
            ),
        ]
    )
