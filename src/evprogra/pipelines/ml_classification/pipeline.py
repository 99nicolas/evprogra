"""
pipeline.py — ml_classification
=================================
4 nodos:
  1. entrenar_random_forest_clf
  2. entrenar_logistic_regression
  3. entrenar_gradient_boosting_clf
  4. consolidar_metricas_clf
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    entrenar_random_forest_clf,
    entrenar_logistic_regression,
    entrenar_gradient_boosting_clf,
    consolidar_metricas_clf,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de clasificación supervisada."""
    return pipeline(
        [
            # ── Nodo 1: RandomForest ─────────────────────────────────────
            node(
                func=entrenar_random_forest_clf,
                inputs=[
                    "X_train_clf", "X_test_clf",
                    "y_train_clf", "y_test_clf",
                    "params:classification.random_forest",
                    "params:classification.cross_validation.cv",
                ],
                outputs="modelo_rf_clf",
                name="node_entrenar_rf_clf",
            ),
            # ── Nodo 2: LogisticRegression ───────────────────────────────
            node(
                func=entrenar_logistic_regression,
                inputs=[
                    "X_train_clf", "X_test_clf",
                    "y_train_clf", "y_test_clf",
                    "params:classification.logistic_regression",
                    "params:classification.cross_validation.cv",
                ],
                outputs="modelo_lr_clf",
                name="node_entrenar_lr_clf",
            ),
            # ── Nodo 3: GradientBoosting ─────────────────────────────────
            node(
                func=entrenar_gradient_boosting_clf,
                inputs=[
                    "X_train_clf", "X_test_clf",
                    "y_train_clf", "y_test_clf",
                    "params:classification.gradient_boosting",
                    "params:classification.cross_validation.cv",
                ],
                outputs="modelo_gbm_clf",
                name="node_entrenar_gbm_clf",
            ),
            # ── Nodo 4: Consolidar métricas + importancias ───────────────
            node(
                func=consolidar_metricas_clf,
                inputs=[
                    "X_train_clf", "X_test_clf",
                    "y_train_clf", "y_test_clf",
                    "modelo_rf_clf",
                    "modelo_lr_clf",
                    "modelo_gbm_clf",
                    "params:classification.cross_validation.cv",
                ],
                outputs=["metricas_clasificacion", "importancias_rf_clf"],
                name="node_metricas_clf",
            ),
        ]
    )
