"""
nodes.py — data_preprocessing pipeline
======================================
Toma rrhh_encoded (output de data_transform de EV1) y genera los splits
de entrenamiento/prueba para clasificación y regresión.

Inputs (desde catalog):
    rrhh_encoded: DataFrame codificado y normalizado (286 x 33)

Outputs (hacia catalog):
    X_train_clf, X_test_clf, y_train_clf, y_test_clf
    X_train_reg, X_test_reg, y_train_reg, y_test_reg
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preparar_datos_clasificacion(
    rrhh_encoded: pd.DataFrame,
    target_clf: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
    stratify_clf: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera el split train/test para la tarea de clasificación.

    Args:
        rrhh_encoded: Dataset integrado y codificado.
        target_clf: Nombre de la variable objetivo (ej: 'segmento_desempeno').
        drop_columns: Columnas a excluir de las features.
        test_size: Proporción del conjunto de prueba (ej: 0.2).
        random_state: Semilla para reproducibilidad.
        stratify_clf: Si True, estratifica el split por clases.

    Returns:
        Tupla (X_train, X_test, y_train, y_test) como DataFrames.
    """
    if target_clf not in rrhh_encoded.columns:
        raise ValueError(f"Variable objetivo '{target_clf}' no encontrada en el dataset.")

    cols_to_drop = [c for c in drop_columns if c in rrhh_encoded.columns]
    X = rrhh_encoded.drop(columns=cols_to_drop)
    y = rrhh_encoded[[target_clf]]

    stratify = rrhh_encoded[target_clf] if stratify_clf else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Clasificación | Train: %d filas | Test: %d filas | Target: %s",
        len(X_train), len(X_test), target_clf
    )
    logger.info("Distribución y_train:\n%s", y_train[target_clf].value_counts().to_dict())

    return X_train, X_test, y_train, y_test


def preparar_datos_regresion(
    rrhh_encoded: pd.DataFrame,
    target_reg: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera el split train/test para la tarea de regresión.

    Args:
        rrhh_encoded: Dataset integrado y codificado.
        target_reg: Nombre de la variable objetivo continua (ej: 'score_desempeno').
        drop_columns: Columnas a excluir de las features.
        test_size: Proporción del conjunto de prueba.
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla (X_train, X_test, y_train, y_test) como DataFrames.
    """
    if target_reg not in rrhh_encoded.columns:
        raise ValueError(f"Variable objetivo '{target_reg}' no encontrada en el dataset.")

    cols_to_drop = [c for c in drop_columns if c in rrhh_encoded.columns]
    X = rrhh_encoded.drop(columns=cols_to_drop)
    y = rrhh_encoded[[target_reg]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(
        "Regresión | Train: %d filas | Test: %d filas | Target: %s",
        len(X_train), len(X_test), target_reg
    )
    logger.info(
        "y_train stats: media=%.3f std=%.3f min=%.3f max=%.3f",
        y_train[target_reg].mean(),
        y_train[target_reg].std(),
        y_train[target_reg].min(),
        y_train[target_reg].max(),
    )

    return X_train, X_test, y_train, y_test
