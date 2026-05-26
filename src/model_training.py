"""
model_training.py
=================
Funciones reutilizables para entrenamiento de modelos supervisados.
Uso:
    from src.model_training import entrenar_clasificadores, entrenar_regresores
"""
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)


def preparar_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: list,
    strategy: str = "median",
) -> tuple:
    """
    Separa features (X) y target (y) con imputación de NaN.

    Args:
        df: DataFrame completo.
        target: Nombre de la columna objetivo.
        drop_cols: Columnas a eliminar junto con el target.
        strategy: Estrategia del SimpleImputer ('median' | 'mean' | 'most_frequent').

    Returns:
        (X_imp, y) — X con NaN imputados, y como Series.
    """
    cols_drop = list(set([c for c in drop_cols if c in df.columns] + [target]))
    X = df.drop(columns=cols_drop).select_dtypes(include=np.number)
    y = df[target]

    imputer = SimpleImputer(strategy=strategy)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    nan_restantes = int(X_imp.isna().sum().sum())
    if nan_restantes > 0:
        logger.warning("Quedan %d NaN en X después de imputar.", nan_restantes)
    else:
        logger.info("Imputación OK — NaN restantes: 0")

    return X_imp, y


def entrenar_clasificadores(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 5,
    models_dir: Path = None,
) -> pd.DataFrame:
    """
    Entrena y evalúa un conjunto de clasificadores.

    Returns:
        DataFrame con métricas por modelo (accuracy, f1, precision, recall, cv_f1).
    """
    modelos = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=42,
        ),
    }

    resultados = []
    for nombre, modelo in modelos.items():
        logger.info("Entrenando %s...", nombre)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test) if hasattr(modelo, "predict_proba") else None
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1_weighted")

        fila = {
            "modelo":      nombre,
            "accuracy":    accuracy_score(y_test, pred),
            "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "precision":   precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall":      recall_score(y_test, pred, average="weighted", zero_division=0),
            "cv_f1_mean":  cv_scores.mean(),
            "cv_f1_std":   cv_scores.std(),
        }
        resultados.append(fila)

        if models_dir:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, Path(models_dir) / f"{nombre.lower()}_clf.pkl")
            logger.info("Modelo guardado: %s_clf.pkl", nombre.lower())

    return pd.DataFrame(resultados).sort_values("f1_weighted", ascending=False)


def entrenar_regresores(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv: int = 5,
    models_dir: Path = None,
) -> pd.DataFrame:
    """
    Entrena y evalúa un conjunto de regresores.

    Returns:
        DataFrame con métricas por modelo (MAE, MSE, RMSE, R², cv_rmse).
    """
    modelos = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0, max_iter=1000),
    }

    resultados = []
    for nombre, modelo in modelos.items():
        logger.info("Entrenando %s (reg)...", nombre)
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

        fila = {
            "modelo":    nombre,
            "mae":       mean_absolute_error(y_test, pred),
            "mse":       mean_squared_error(y_test, pred),
            "rmse":      mean_squared_error(y_test, pred) ** 0.5,
            "r2":        r2_score(y_test, pred),
            "cv_rmse_mean": -cv_scores.mean(),
            "cv_rmse_std":   cv_scores.std(),
        }
        resultados.append(fila)

        if models_dir:
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(modelo, Path(models_dir) / f"{nombre.lower()}_reg.pkl")
            logger.info("Modelo guardado: %s_reg.pkl", nombre.lower())

    return pd.DataFrame(resultados).sort_values("rmse")
