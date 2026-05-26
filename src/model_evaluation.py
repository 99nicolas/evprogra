"""
model_evaluation.py
===================
Funciones reutilizables para evaluación y visualización de modelos.
Uso:
    from src.model_evaluation import plot_confusion_matrix, plot_feature_importance
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, r2_score, mean_absolute_error,
)

logger = logging.getLogger(__name__)


def reporte_clasificacion(
    y_true, y_pred, modelo_nombre: str = "Modelo", save_path: Path = None
) -> pd.DataFrame:
    """
    Genera y opcionalmente guarda el reporte de clasificación como DataFrame.

    Returns:
        DataFrame con precision, recall, f1-score por clase.
    """
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).T
    logger.info("=== Reporte: %s ===\n%s", modelo_nombre, classification_report(y_true, y_pred, zero_division=0))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        logger.info("Reporte guardado en %s", save_path)

    return df


def plot_confusion_matrix(
    y_true, y_pred, classes, titulo: str = "Matriz de Confusión",
    save_path: Path = None
) -> None:
    """Grafica la matriz de confusión y la guarda si se indica ruta."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titulo)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        logger.info("Gráfico guardado: %s", save_path)
    plt.show()


def plot_feature_importance(
    model, feature_names, top_n: int = 15,
    titulo: str = "Importancia de Variables",
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Grafica y retorna la importancia de variables de un modelo árbol.

    Args:
        model: Modelo con atributo feature_importances_.
        feature_names: Lista o Index con nombres de columnas.
        top_n: Número de variables a mostrar.
        titulo: Título del gráfico.
        save_path: Ruta donde guardar la imagen (opcional).

    Returns:
        DataFrame con columnas ['feature', 'importance'].
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("El modelo no tiene atributo 'feature_importances_'.")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    sns.barplot(data=imp_df, y="feature", x="importance", palette="crest", ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Importancia relativa")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        logger.info("Gráfico guardado: %s", save_path)
    plt.show()

    return imp_df


def plot_pred_vs_real(
    y_true, y_pred,
    titulo: str = "Predicción vs Real",
    save_path: Path = None,
) -> dict:
    """
    Scatter plot de valores predichos vs reales en regresión.

    Returns:
        dict con 'rmse' y 'r2'.
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.set_title(f"{titulo}\nRMSE={rmse:.3f}  R²={r2:.3f}  MAE={mae:.3f}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        logger.info("Gráfico guardado: %s", save_path)
    plt.show()

    return {"rmse": rmse, "r2": r2, "mae": mae}


def comparar_modelos(resultados_df: pd.DataFrame, metrica: str, titulo: str = "",
                     save_path: Path = None) -> None:
    """Gráfico de barras comparando modelos por una métrica dada."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=resultados_df, x="modelo", y=metrica, palette="viridis", ax=ax)
    ax.set_title(titulo or f"Comparación de modelos — {metrica}")
    ax.set_ylabel(metrica)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()
