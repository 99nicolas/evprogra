"""
nodes.py — ml_optimization pipeline
=====================================
Optimiza hiperparámetros de dos modelos de clasificación:
  1. RandomForestClassifier → GridSearchCV
  2. GradientBoostingClassifier → RandomizedSearchCV

Inputs (desde catalog):
    X_train_clf, X_test_clf, y_train_clf, y_test_clf

Outputs (hacia catalog):
    modelo_rf_clf_optimizado
    modelo_gbm_clf_optimizado
    metricas_optimizacion
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def _preparar_xy(X: pd.DataFrame, y: pd.DataFrame, imputer=None):
    """Retorna solo features numéricas, imputa NaNs y target como vector 1D."""
    X_num = X.select_dtypes(include=[np.number])
    
    # Imputa valores NaN con mediana si no se proporciona imputer
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_num)
    else:
        X_imputed = imputer.transform(X_num)
    
    X_imputed = pd.DataFrame(X_imputed, columns=X_num.columns)
    y_arr = y.iloc[:, 0].values
    return X_imputed, y_arr, imputer


def _metricas_modelo(nombre: str, modelo, X_test, y_test) -> dict:
    """Calcula métricas estándar de clasificación multiclase."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test) if hasattr(modelo, "predict_proba") else None

    out = {
        "modelo": nombre,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

    if y_proba is not None:
        try:
            out["roc_auc_weighted"] = round(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4
            )
        except Exception:
            out["roc_auc_weighted"] = None
    else:
        out["roc_auc_weighted"] = None

    return out


def optimizar_random_forest_clf(
    X_train_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    params_grid: dict,
    cv: int,
    scoring: str,
    n_jobs: int,
    verbose: int,
) -> RandomForestClassifier:
    """
    Optimiza RandomForestClassifier con GridSearchCV.

    Args:
        X_train_clf: Features de entrenamiento.
        y_train_clf: Target de entrenamiento.
        params_grid: Espacio de búsqueda desde parameters.yml.
        cv: Número de folds.
        scoring: Métrica objetivo (ej: f1_weighted).
        n_jobs: Núcleos paralelos.
        verbose: Nivel de logging de sklearn.

    Returns:
        Mejor estimador encontrado por GridSearchCV.
    """
    X_tr, y_tr, _ = _preparar_xy(X_train_clf, y_train_clf)

    base_model = RandomForestClassifier(random_state=42)
    search = GridSearchCV(
        estimator=base_model,
        param_grid=params_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    logger.info("GridSearchCV RF mejor score: %.4f", search.best_score_)
    logger.info("GridSearchCV RF mejores params: %s", search.best_params_)
    return search.best_estimator_


def optimizar_gbm_clf(
    X_train_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    random_search_cfg: dict,
    random_state: int,
    cv: int,
    scoring: str,
    n_jobs: int,
    verbose: int,
) -> GradientBoostingClassifier:
    """
    Optimiza GradientBoostingClassifier con RandomizedSearchCV.

    Args:
        X_train_clf: Features de entrenamiento.
        y_train_clf: Target de entrenamiento.
        random_search_cfg: Configuración con n_iter y param_distributions.
        random_state: Semilla para reproducibilidad.
        cv: Número de folds.
        scoring: Métrica objetivo.
        n_jobs: Núcleos paralelos.
        verbose: Nivel de logging de sklearn.

    Returns:
        Mejor estimador encontrado por RandomizedSearchCV.
    """
    X_tr, y_tr, _ = _preparar_xy(X_train_clf, y_train_clf)

    base_model = GradientBoostingClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=random_search_cfg["param_distributions"],
        n_iter=random_search_cfg["n_iter"],
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_tr, y_tr)

    logger.info("RandomizedSearchCV GBM mejor score: %.4f", search.best_score_)
    logger.info("RandomizedSearchCV GBM mejores params: %s", search.best_params_)
    return search.best_estimator_


def consolidar_metricas_optimizacion(
    X_test_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    modelo_rf_clf_optimizado: RandomForestClassifier,
    modelo_gbm_clf_optimizado: GradientBoostingClassifier,
) -> pd.DataFrame:
    """
    Consolida métricas comparativas de los modelos optimizados.

    Returns:
        DataFrame con accuracy, F1, precision, recall y ROC-AUC.
    """
    X_te, y_te, _ = _preparar_xy(X_test_clf, y_test_clf)

    resultados = [
        _metricas_modelo("RandomForest_Optimizado", modelo_rf_clf_optimizado, X_te, y_te),
        _metricas_modelo("GradientBoosting_Optimizado", modelo_gbm_clf_optimizado, X_te, y_te),
    ]

    metricas_df = pd.DataFrame(resultados)
    logger.info("Métricas optimización:%s", metricas_df.to_string(index=False))
    return metricas_df
