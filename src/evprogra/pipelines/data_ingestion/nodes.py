"""
Pipeline: data_ingestion
AD 1.1 — Carga y exploración inicial de los 4 datasets RRHH
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _explorar_dataset(df: pd.DataFrame, nombre: str) -> dict:
    """Genera métricas exploratorias de un DataFrame."""
    nulos = df.isnull().sum()
    return {
        "dataset": nombre,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_nombres": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulos_por_columna": nulos.to_dict(),
        "total_nulos": int(nulos.sum()),
        "porcentaje_nulos": round(nulos.sum() / df.size * 100, 2),
        "duplicados": int(df.duplicated().sum()),
        "describe": df.describe(include="all").to_dict(),
    }


def _detectar_problemas(info: dict) -> list[dict]:
    """Detecta problemas de calidad a partir de las métricas exploratorias."""
    problemas = []
    nombre = info["dataset"]

    if info["duplicados"] > 0:
        problemas.append({
            "dataset": nombre,
            "tipo": "DUPLICADOS",
            "severidad": "WARN",
            "detalle": f"{info['duplicados']} filas duplicadas",
        })

    for col, n_nulos in info["nulos_por_columna"].items():
        if n_nulos == 0:
            continue
        pct = round(n_nulos / info["filas"] * 100, 2)
        sev = "FAIL" if pct > 30 else "WARN"
        problemas.append({
            "dataset": nombre,
            "tipo": "NULOS",
            "severidad": sev,
            "detalle": f"'{col}': {n_nulos} nulos ({pct}%)",
        })

    for col, dtype in info["dtypes"].items():
        if "object" in dtype:
            problemas.append({
                "dataset": nombre,
                "tipo": "TIPO_OBJECT",
                "severidad": "INFO",
                "detalle": f"'{col}' es tipo object — revisar si corresponde numérico/fecha",
            })

    return problemas


# ── nodos públicos ────────────────────────────────────────────────────────────

def cargar_y_explorar_datasets(
    empleados: pd.DataFrame,
    ausencias: pd.DataFrame,
    capacitaciones: pd.DataFrame,
    evaluaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nodo principal del pipeline data_ingestion.
    Recibe los 4 DataFrames crudos, genera exploración y reporte de diagnóstico.
    """
    datasets = {
        "empleados": empleados,
        "ausencias": ausencias,
        "capacitaciones": capacitaciones,
        "evaluaciones": evaluaciones,
    }

    filas_reporte = []

    for nombre, df in datasets.items():
        info = _explorar_dataset(df, nombre)
        problemas = _detectar_problemas(info)

        logger.info(
            "Dataset '%s' | shape=%s | nulos=%d | duplicados=%d | problemas=%d",
            nombre,
            df.shape,
            info["total_nulos"],
            info["duplicados"],
            len(problemas),
        )

        # Fila de resumen del dataset
        filas_reporte.append({
            "dataset": nombre,
            "tipo_check": "RESUMEN",
            "severidad": "INFO",
            "filas": info["filas"],
            "columnas": info["columnas"],
            "total_nulos": info["total_nulos"],
            "porcentaje_nulos": info["porcentaje_nulos"],
            "duplicados": info["duplicados"],
            "detalle": f"shape=({info['filas']}, {info['columnas']})",
        })

        # Filas de problemas detectados
        for p in problemas:
            filas_reporte.append({
                "dataset": p["dataset"],
                "tipo_check": p["tipo"],
                "severidad": p["severidad"],
                "filas": info["filas"],
                "columnas": info["columnas"],
                "total_nulos": info["total_nulos"],
                "porcentaje_nulos": info["porcentaje_nulos"],
                "duplicados": info["duplicados"],
                "detalle": p["detalle"],
            })

    reporte = pd.DataFrame(filas_reporte)

    n_fail = (reporte["severidad"] == "FAIL").sum()
    n_warn = (reporte["severidad"] == "WARN").sum()
    n_info = (reporte["severidad"] == "INFO").sum()

    logger.info(
        "Reporte diagnóstico: %d checks | FAIL=%d WARN=%d INFO=%d",
        len(reporte), n_fail, n_warn, n_info,
    )

    return reporte
