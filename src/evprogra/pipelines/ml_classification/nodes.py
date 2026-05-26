"""
nodes.py — ml_classification pipeline
======================================
Entrena y evalúa 3 modelos de clasificación sobre segmento_desempeno:
  - RandomForestClassifier
  - LogisticRegression
  - GradientBoostingClassifier

Inputs (desde catalog):
    X_train_clf, X_test_clf, y_train_clf, y_test_clf

Outputs (hacia catalog):
    modelo_rf_clf, modelo_lr_clf, modelo_gbm_clf
    metricas_clasificacion, importancias_rf_clf
"""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
)

logger = logging.getLogger(__name__)


def _preparar_xy(
    X: pd.DataFrame,
    y: pd.DataFrame,
    imputer: SimpleImputer = None,
    fit: bool = True,
) -> tuple:
    """
    Convierte DataFrames a arrays listos para sklearn.
    Imputa NaN con mediana para evitar errores en estimadores
    que no aceptan valores faltantes (GradientBoosting, LogisticRegression).

    Args:
        X: DataFrame de features.
        y: DataFrame con la columna target.
        imputer: Instancia de SimpleImputer ya ajustada (para test set).
                 Si es None se crea uno nuevo.
        fit: True → fit_transform (usar en train).
             False → solo transform (usar en test, pasar imputer ajustado).

    Returns:
        (X_imputado, y_array, imputer) — el imputer se retorna para
        reutilizarlo en el conjunto de test.
    """
    X_num = X.select_dtypes(include=[np.number])
    y_arr = y.iloc[:, 0].values

    if imputer is None:
        imputer = SimpleImputer(strategy="median")

    if fit:
        X_imp = imputer.fit_transform(X_num)
    else:
        X_imp = imputer.transform(X_num)

    nan_restantes = np.isnan(X_imp).sum()
    if nan_restantes > 0:
        logger.warning("_preparar_xy: quedan %d NaN después de imputar.", nan_restantes)
    else:
        logger.info("_preparar_xy: imputación OK — 0 NaN (fit=%s)", fit)

    return pd.DataFrame(X_imp, columns=X_num.columns), y_arr, imputer


def _calcular_metricas(
    nombre: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    cv_scores: np.ndarray,
    clases: list,
) -> dict:
    """Calcula accuracy, F1, precision, recall, ROC-AUC y CV score."""
    metricas = {
        "modelo": nombre,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "n_clases": len(clases),
    }
    if y_proba is not None:
        try:
            auc = roc_auc_score(
                y_true, y_proba,
                multi_class="ovr", average="weighted",
            )
            metricas["roc_auc_weighted"] = round(auc, 4)
        except Exception:
            metricas["roc_auc_weighted"] = None
    else:
        metricas["roc_auc_weighted"] = None

    logger.info("── %s ──", nombre)
    logger.info("  Accuracy : %.4f", metricas["accuracy"])
    logger.info("  F1 (w)   : %.4f", metricas["f1_weighted"])
    logger.info("  ROC-AUC  : %s",   metricas["roc_auc_weighted"])
    logger.info("  CV F1    : %.4f ± %.4f", metricas["cv_f1_mean"], metricas["cv_f1_std"])
    logger.info(classification_report(y_true, y_pred, zero_division=0))

    return metricas


def entrenar_random_forest_clf(
    X_train_clf: pd.DataFrame,
    X_test_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    params_rf: dict,
    cv: int,
) -> RandomForestClassifier:
    """
    Entrena un RandomForestClassifier.

    Args:
        X_train_clf: Features de entrenamiento.
        X_test_clf: Features de prueba.
        y_train_clf: Labels de entrenamiento.
        y_test_clf: Labels de prueba.
        params_rf: Hiperparámetros desde parameters.yml (classification.random_forest).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy(X_train_clf, y_train_clf, fit=True)
    X_te, y_te, _   = _preparar_xy(X_test_clf,  y_test_clf,  imputer=imp, fit=False)

    modelo = RandomForestClassifier(**params_rf)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    y_proba   = modelo.predict_proba(X_te)
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted")

    _calcular_metricas(
        "RandomForest", y_te, y_pred, y_proba,
        cv_scores, list(modelo.classes_)
    )
    return modelo


def entrenar_logistic_regression(
    X_train_clf: pd.DataFrame,
    X_test_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    params_lr: dict,
    cv: int,
) -> LogisticRegression:
    """
    Entrena una LogisticRegression multiclase.

    Args:
        X_train_clf: Features de entrenamiento.
        X_test_clf: Features de prueba.
        y_train_clf: Labels de entrenamiento.
        y_test_clf: Labels de prueba.
        params_lr: Hiperparámetros desde parameters.yml (classification.logistic_regression).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy(X_train_clf, y_train_clf, fit=True)
    X_te, y_te, _   = _preparar_xy(X_test_clf,  y_test_clf,  imputer=imp, fit=False)

    modelo = LogisticRegression(**params_lr)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    y_proba   = modelo.predict_proba(X_te)
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted")

    _calcular_metricas(
        "LogisticRegression", y_te, y_pred, y_proba,
        cv_scores, list(modelo.classes_)
    )
    return modelo


def entrenar_gradient_boosting_clf(
    X_train_clf: pd.DataFrame,
    X_test_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    params_gbm: dict,
    cv: int,
) -> GradientBoostingClassifier:
    """
    Entrena un GradientBoostingClassifier.

    Args:
        X_train_clf: Features de entrenamiento.
        X_test_clf: Features de prueba.
        y_train_clf: Labels de entrenamiento.
        y_test_clf: Labels de prueba.
        params_gbm: Hiperparámetros desde parameters.yml (classification.gradient_boosting).
        cv: Número de folds para validación cruzada.

    Returns:
        Modelo entrenado.
    """
    X_tr, y_tr, imp = _preparar_xy(X_train_clf, y_train_clf, fit=True)
    X_te, y_te, _   = _preparar_xy(X_test_clf,  y_test_clf,  imputer=imp, fit=False)

    modelo = GradientBoostingClassifier(**params_gbm)
    modelo.fit(X_tr, y_tr)

    y_pred    = modelo.predict(X_te)
    y_proba   = modelo.predict_proba(X_te)
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted")

    _calcular_metricas(
        "GradientBoosting", y_te, y_pred, y_proba,
        cv_scores, list(modelo.classes_)
    )
    return modelo


def consolidar_metricas_clf(
    X_train_clf: pd.DataFrame,
    X_test_clf: pd.DataFrame,
    y_train_clf: pd.DataFrame,
    y_test_clf: pd.DataFrame,
    modelo_rf_clf: RandomForestClassifier,
    modelo_lr_clf: LogisticRegression,
    modelo_gbm_clf: GradientBoostingClassifier,
    cv: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida métricas comparativas y feature importances del RF.

    Returns:
        Tupla (metricas_clasificacion, importancias_rf_clf) como DataFrames.
    """
    X_tr, y_tr, imp = _preparar_xy(X_train_clf, y_train_clf, fit=True)
    X_te, y_te, _   = _preparar_xy(X_test_clf,  y_test_clf,  imputer=imp, fit=False)

    resultados = []
    for nombre, modelo in [
        ("RandomForest",      modelo_rf_clf),
        ("LogisticRegression", modelo_lr_clf),
        ("GradientBoosting",  modelo_gbm_clf),
    ]:
        y_pred    = modelo.predict(X_te)
        y_proba   = modelo.predict_proba(X_te) if hasattr(modelo, "predict_proba") else None
        cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="f1_weighted")
        m = _calcular_metricas(
            nombre, y_te, y_pred, y_proba,
            cv_scores, list(modelo.classes_)
        )
        resultados.append(m)

    metricas_df = pd.DataFrame(resultados)

    # Feature importances RF
    feature_names = X_train_clf.select_dtypes(include=[np.number]).columns.tolist()
    importancias_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": modelo_rf_clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Top 10 features RF clasificación:")
    logger.info(importancias_df.head(10).to_string())

    return metricas_df, importancias_df