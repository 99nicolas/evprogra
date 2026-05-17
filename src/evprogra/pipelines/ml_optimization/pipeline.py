"""
pipeline.py — ml_optimization
===============================
3 nodos:
  1. optimizar_random_forest_clf  (GridSearchCV)
  2. optimizar_gbm_clf            (RandomizedSearchCV)
  3. consolidar_metricas_optimizacion
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    optimizar_random_forest_clf,
    optimizar_gbm_clf,
    consolidar_metricas_optimizacion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de optimización de hiperparámetros."""
    return pipeline(
        [
            node(
                func=optimizar_random_forest_clf,
                inputs=[
                    "X_train_clf",
                    "y_train_clf",
                    "params:hyperparameter_tuning.grid_search_rf_clf",
                    "params:hyperparameter_tuning.cv",
                    "params:hyperparameter_tuning.scoring_clf",
                    "params:hyperparameter_tuning.n_jobs",
                    "params:hyperparameter_tuning.verbose",
                ],
                outputs="modelo_rf_clf_optimizado",
                name="node_gridsearch_rf_clf",
            ),
            node(
                func=optimizar_gbm_clf,
                inputs=[
                    "X_train_clf",
                    "y_train_clf",
                    "params:hyperparameter_tuning.random_search_gbm_clf",
                    "params:hyperparameter_tuning.random_state",
                    "params:hyperparameter_tuning.cv",
                    "params:hyperparameter_tuning.scoring_clf",
                    "params:hyperparameter_tuning.n_jobs",
                    "params:hyperparameter_tuning.verbose",
                ],
                outputs="modelo_gbm_clf_optimizado",
                name="node_randomsearch_gbm_clf",
            ),
            node(
                func=consolidar_metricas_optimizacion,
                inputs=[
                    "X_test_clf",
                    "y_test_clf",
                    "modelo_rf_clf_optimizado",
                    "modelo_gbm_clf_optimizado",
                ],
                outputs="metricas_optimizacion",
                name="node_metricas_optimizacion",
            ),
        ]
    )
