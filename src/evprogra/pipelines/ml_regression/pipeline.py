"""
pipeline.py — ml_regression
=============================
4 nodos:
  1. entrenar_random_forest_reg
  2. entrenar_ridge
  3. entrenar_gradient_boosting_reg
  4. consolidar_metricas_reg
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    entrenar_random_forest_reg,
    entrenar_ridge,
    entrenar_gradient_boosting_reg,
    consolidar_metricas_reg,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de regresión supervisada."""
    return pipeline(
        [
            # ── Nodo 1: RandomForest ─────────────────────────────────────
            node(
                func=entrenar_random_forest_reg,
                inputs=[
                    "X_train_reg", "X_test_reg",
                    "y_train_reg", "y_test_reg",
                    "params:regression.random_forest",
                    "params:regression.cross_validation.cv",
                ],
                outputs="modelo_rf_reg",
                name="node_entrenar_rf_reg",
            ),
            # ── Nodo 2: Ridge ────────────────────────────────────────────
            node(
                func=entrenar_ridge,
                inputs=[
                    "X_train_reg", "X_test_reg",
                    "y_train_reg", "y_test_reg",
                    "params:regression.ridge",
                    "params:regression.cross_validation.cv",
                ],
                outputs="modelo_ridge_reg",
                name="node_entrenar_ridge",
            ),
            # ── Nodo 3: GradientBoosting ─────────────────────────────────
            node(
                func=entrenar_gradient_boosting_reg,
                inputs=[
                    "X_train_reg", "X_test_reg",
                    "y_train_reg", "y_test_reg",
                    "params:regression.gradient_boosting",
                    "params:regression.cross_validation.cv",
                ],
                outputs="modelo_gbm_reg",
                name="node_entrenar_gbm_reg",
            ),
            # ── Nodo 4: Consolidar métricas + importancias ───────────────
            node(
                func=consolidar_metricas_reg,
                inputs=[
                    "X_train_reg", "X_test_reg",
                    "y_train_reg", "y_test_reg",
                    "modelo_rf_reg",
                    "modelo_ridge_reg",
                    "modelo_gbm_reg",
                    "params:regression.cross_validation.cv",
                ],
                outputs=["metricas_regresion", "importancias_rf_reg"],
                name="node_metricas_reg",
            ),
        ]
    )
