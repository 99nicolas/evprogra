"""
nodes.py — ml_regression pipeline
====================================
Entrena y evalúa 3 modelos de regresión sobre score_desempeno (continua [0,7]):
  - RandomForestRegressor
  - Ridge
  - GradientBoostingRegressor

Inputs (desde catalog):
    X_train_reg, X_test_reg, y_train_reg, y_test_reg

Outputs (hacia catalog):
    modelo_rf_reg, modelo_ridge_reg, modelo_gbm_reg
    metricas_regresion, importancias_rf_reg
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def _preparar_xy_reg(
    X: pd.DataFrame,
    y: pd.DataFrame,
    imputer: SimpleImputer = None,
    fit: bool = True,
) -> tuple:
    """
    Convierte DataFrames a arrays numpy para sklearn, imputando NaN con mediana.

    Args:
        X: DataFrame de features.
        y: DataFrame con la columna target.
        imputer: Instancia de SimpleImputer ya ajustada (para test set).
                 Si es None se crea uno nuevo.
        fit: True → fit_transform (usar en train).
             False → solo transform (usar en test, pasar imputer ajustado).

    Returns:
        (X_imputado, y_array, imputer)
    """
    X_num = X.select_dtypes(include=[np.number])
    y_arr = y.iloc[:, 0].values.astype(float)

    if imputer is None:
        imputer = SimpleImputer(strategy="median")

    if fit:
        X_imp = imputer.fit_transform(X_num)
    else:
        X_imp = imputer.transform(X_num)

    nan_restantes = np.isnan(X_imp).sum()
    if nan_restantes > 0:
        logger.warning("_preparar_xy_reg: quedan %d NaN después de imputar.", nan_restantes)
    else:
        logger.info("_preparar_xy_reg: imputación OK — 0 NaN (fit=%s)", fit)

    return pd.DataFrame(X_imp, columns=X_num.columns), y_arr, imputer


def _calcular_metricas_reg(
    nombre: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_scores: np.ndarray,
) -> dict:
    """Calcula MAE, MSE, RMSE, R² y CV RMSE."""
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metricas = {
        "modelo":       nombre,
        "mae":          round(mae,  4),
        "mse":          round(mse,  4),
        "rmse":         round(rmse, 4),
        "r2":           round(r2,   4),
        "cv_rmse_mean": round(-cv_scores.mean(), 4),
        "cv_rmse_std":  round(cv_scores.std(),   4),
    }

    logger.info("── %s ──", nombre)
    logger.info("  MAE  : %.4f", mae)
    logger.info("  RMSE : %.4f", rmse)
    logger.info("  R²   : %.4f", r2)
    logger.info("  CV RMSE: %.4f ± %.4f", -cv_scores.mean(), cv_scores.std())
    return metricas


def entrenar_random_forest_reg(
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    params_rf: dict,
    cv: int,
) -> RandomForestRegressor:
    """
    Entrena un RandomForestRegressor.

    Args:
        X_train_reg: Features de entrenamiento para regresión.
        X_test_reg: Features de prueba para regresión.
        y_train_reg: Variable objetivo de entrenamiento (score_desempeno).
        y_test_reg: Variable objetivo de prueba.
        params_rf: Hiperparámetros desde parameters.yml (regression.random_forest).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo RandomForestRegressor entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy_reg(X_train_reg, y_train_reg, fit=True)
    X_te, y_te, _   = _preparar_xy_reg(X_test_reg,  y_test_reg,  imputer=imp, fit=False)

    modelo = RandomForestRegressor(**params_rf)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    cv_scores = cross_val_score(
        modelo, X_tr, y_tr, cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    _calcular_metricas_reg("RandomForest", y_te, y_pred, cv_scores)
    return modelo


def entrenar_ridge(
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    params_ridge: dict,
    cv: int,
) -> Ridge:
    """
    Entrena un modelo Ridge (regresión lineal regularizada L2).

    Args:
        X_train_reg: Features de entrenamiento.
        X_test_reg: Features de prueba.
        y_train_reg: Variable objetivo de entrenamiento.
        y_test_reg: Variable objetivo de prueba.
        params_ridge: Hiperparámetros desde parameters.yml (regression.ridge).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo Ridge entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy_reg(X_train_reg, y_train_reg, fit=True)
    X_te, y_te, _   = _preparar_xy_reg(X_test_reg,  y_test_reg,  imputer=imp, fit=False)

    modelo = Ridge(**params_ridge)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    cv_scores = cross_val_score(
        modelo, X_tr, y_tr, cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    _calcular_metricas_reg("Ridge", y_te, y_pred, cv_scores)
    return modelo


def entrenar_gradient_boosting_reg(
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    params_gbm: dict,
    cv: int,
) -> GradientBoostingRegressor:
    """
    Entrena un GradientBoostingRegressor.

    Args:
        X_train_reg: Features de entrenamiento.
        X_test_reg: Features de prueba.
        y_train_reg: Variable objetivo de entrenamiento.
        y_test_reg: Variable objetivo de prueba.
        params_gbm: Hiperparámetros desde parameters.yml (regression.gradient_boosting).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo GradientBoostingRegressor entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy_reg(X_train_reg, y_train_reg, fit=True)
    X_te, y_te, _   = _preparar_xy_reg(X_test_reg,  y_test_reg,  imputer=imp, fit=False)

    modelo = GradientBoostingRegressor(**params_gbm)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    cv_scores = cross_val_score(
        modelo, X_tr, y_tr, cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    _calcular_metricas_reg("GradientBoosting", y_te, y_pred, cv_scores)
    return modelo


def consolidar_metricas_reg(
    X_train_reg: pd.DataFrame,
    X_test_reg: pd.DataFrame,
    y_train_reg: pd.DataFrame,
    y_test_reg: pd.DataFrame,
    modelo_rf_reg: RandomForestRegressor,
    modelo_ridge_reg: Ridge,
    modelo_gbm_reg: GradientBoostingRegressor,
    cv: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida tabla comparativa de métricas y feature importances del RF.

    Returns:
        Tupla (metricas_regresion, importancias_rf_reg) como DataFrames.
    """
    X_tr, y_tr, imp = _preparar_xy_reg(X_train_reg, y_train_reg, fit=True)
    X_te, y_te, _   = _preparar_xy_reg(X_test_reg,  y_test_reg,  imputer=imp, fit=False)

    resultados = []
    for nombre, modelo in [
        ("RandomForest",     modelo_rf_reg),
        ("Ridge",            modelo_ridge_reg),
        ("GradientBoosting", modelo_gbm_reg),
    ]:
        y_pred    = modelo.predict(X_te)
        cv_scores = cross_val_score(
            modelo, X_tr, y_tr, cv=cv,
            scoring="neg_root_mean_squared_error"
        )
        m = _calcular_metricas_reg(nombre, y_te, y_pred, cv_scores)
        resultados.append(m)

    metricas_df = pd.DataFrame(resultados)

    # Feature importances del RandomForest
    feature_names = X_train_reg.select_dtypes(include=[np.number]).columns.tolist()
    importancias_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": modelo_rf_reg.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Top 10 features RF regresión:")
    logger.info(importancias_df.head(10).to_string())

    return metricas_df, importancias_df
