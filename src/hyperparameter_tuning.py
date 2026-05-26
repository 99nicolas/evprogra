"""
hyperparameter_tuning.py
========================
Funciones reutilizables para búsqueda de hiperparámetros con GridSearchCV
y RandomizedSearchCV.

Uso:
    from src.hyperparameter_tuning import grid_search_rf, random_search_gbm
"""
import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Parámetros por defecto
# ──────────────────────────────────────────────────────────

PARAM_GRID_RF = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "class_weight":    ["balanced"],
}

PARAM_DIST_GBM = {
    "n_estimators":   [50, 100, 150, 200, 300],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2],
    "max_depth":      [3, 4, 5, 6, 8],
    "subsample":      [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_samples_split": [2, 5, 10],
}

PARAM_GRID_RIDGE = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
}

PARAM_DIST_GBM_REG = {
    "n_estimators":  [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 4, 5],
    "subsample":     [0.7, 0.8, 1.0],
}


# ──────────────────────────────────────────────────────────
# Funciones públicas
# ──────────────────────────────────────────────────────────

def grid_search_rf(
    X_train, y_train,
    param_grid: dict = None,
    cv: int = 5,
    scoring: str = "f1_weighted",
    save_path: Path = None,
) -> GridSearchCV:
    """
    GridSearchCV sobre RandomForestClassifier.

    Returns:
        GridSearchCV ajustado. Accede al mejor modelo con .best_estimator_.
    """
    pg = param_grid or PARAM_GRID_RF
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        pg, cv=cv, scoring=scoring, n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    logger.info("RF GridSearch — Mejor %s: %.4f", scoring, gs.best_score_)
    logger.info("Mejores parámetros RF: %s", gs.best_params_)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, save_path)
        logger.info("Modelo guardado: %s", save_path)

    return gs


def random_search_gbm(
    X_train, y_train,
    param_distributions: dict = None,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "f1_weighted",
    save_path: Path = None,
) -> RandomizedSearchCV:
    """
    RandomizedSearchCV sobre GradientBoostingClassifier.

    Returns:
        RandomizedSearchCV ajustado.
    """
    pd_ = param_distributions or PARAM_DIST_GBM
    rs = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        pd_, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=-1, random_state=42, verbose=1,
    )
    rs.fit(X_train, y_train)
    logger.info("GBM RandomSearch — Mejor %s: %.4f", scoring, rs.best_score_)
    logger.info("Mejores parámetros GBM: %s", rs.best_params_)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rs.best_estimator_, save_path)
        logger.info("Modelo guardado: %s", save_path)

    return rs


def grid_search_ridge(
    X_train, y_train,
    param_grid: dict = None,
    cv: int = 5,
    save_path: Path = None,
) -> GridSearchCV:
    """GridSearchCV sobre Ridge para regresión."""
    pg = param_grid or PARAM_GRID_RIDGE
    gs = GridSearchCV(
        Ridge(), pg, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    logger.info("Ridge GridSearch — Mejor alpha: %s", gs.best_params_)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, save_path)

    return gs


def random_search_gbm_reg(
    X_train, y_train,
    param_distributions: dict = None,
    n_iter: int = 20,
    cv: int = 5,
    save_path: Path = None,
) -> RandomizedSearchCV:
    """RandomizedSearchCV sobre GradientBoostingRegressor."""
    pd_ = param_distributions or PARAM_DIST_GBM_REG
    rs = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        pd_, n_iter=n_iter, cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, random_state=42,
    )
    rs.fit(X_train, y_train)
    logger.info("GBM Reg RandomSearch — Mejor RMSE CV: %.4f", -rs.best_score_)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rs.best_estimator_, save_path)

    return rs


def consolidar_resultados_optimizacion(
    resultados: list[dict], save_path: Path = None
) -> pd.DataFrame:
    """
    Consolida métricas de múltiples búsquedas en un DataFrame.

    Args:
        resultados: Lista de dicts con claves 'modelo', 'mejor_score', 'mejores_params'.
        save_path: Ruta CSV para guardar.
    """
    df = pd.DataFrame(resultados)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Consolidado guardado: %s", save_path)
    return df
