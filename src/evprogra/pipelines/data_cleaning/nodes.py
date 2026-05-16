"""
nodes.py — Pipeline: data_cleaning
====================================
Funciones de limpieza para cada dataset de RRHH.
Cada función recibe el DataFrame crudo y el bloque 'cleaning' de parameters.yml.
Aplica:
  - Normalización de strings (strip, title case)
  - Parsing de fechas con formatos mixtos
  - Imputación de nulos (mediana para numéricos, moda para categóricos)
  - Eliminación de duplicados
  - Tratamiento de outliers (IQR o Z-score)
  - Corrección de tipos de datos
"""


import logging
import re
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)



# =============================================================================
# UTILIDADES COMPARTIDAS
# =============================================================================


def _limpiar_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Recodifica strings a cp1252-safe para evitar UnicodeEncodeError en Windows."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.encode("cp1252", errors="replace").decode("cp1252")
            if isinstance(x, str) else x
        )
    return df


def _parse_dates_mixed(series: pd.Series) -> pd.Series:
    """
    Parsea fechas con formatos mixtos: 'YYYY-MM-DD' y 'DD/MM/YYYY'.
    También trunca timestamps largos (YYYY-MM-DD HH:MM:SS...) a solo fecha.
    """
    s = series.astype(str).str[:10].replace("nan", np.nan)
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    mask = parsed.isna() & s.notna()
    parsed[mask] = pd.to_datetime(s[mask], format="%d/%m/%Y", errors="coerce")
    return parsed



def _normalize_text(series: pd.Series) -> pd.Series:
    """Elimina espacios extra y unifica a Title Case."""
    return series.astype(str).str.strip().str.title().replace("Nan", np.nan)



def _remove_special_chars(series: pd.Series) -> pd.Series:
    """Elimina caracteres especiales en columnas numéricas ($, ~, espacios)."""
    return (
        series.astype(str)
        .str.replace(r"[\$~\s]", "", regex=True)
        .replace("nan", np.nan)
    )



def _treat_outliers_iqr(
    df: pd.DataFrame, col: str, factor: float = 1.5
) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    logger.info(
        "Outliers IQR en '%s': %d valores cappados [%.2f, %.2f]",
        col, outliers_count, lower, upper,
    )
    return df



def _treat_outliers_zscore(
    df: pd.DataFrame, col: str, threshold: float = 3.0
) -> pd.DataFrame:
    col_data = df[col].dropna()
    mean = col_data.mean()
    std = col_data.std()
    if std == 0:
        return df
    z = np.abs((col_data - mean) / std)
    outlier_idx = col_data.index[z > threshold]
    df.loc[outlier_idx, col] = np.nan
    df[col] = df[col].fillna(df[col].median())
    logger.info("Outliers Z-score en '%s': %d valores tratados", col, len(outlier_idx))
    return df



def _apply_outliers(df: pd.DataFrame, cols: list, params: dict) -> pd.DataFrame:
    method = params.get("outlier_method", "iqr")
    factor = params.get("iqr_factor", 1.5)
    for col in cols:
        if col in df.columns:
            if method == "iqr":
                df = _treat_outliers_iqr(df, col, factor)
            else:
                df = _treat_outliers_zscore(df, col)
    return df



def _impute_nulls(df: pd.DataFrame, median_cols: list, mode_cols: list) -> pd.DataFrame:
    for col in median_cols:
        if col in df.columns and df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.info("Imputado '%s' con mediana=%.2f", col, med)

    for col in mode_cols:
        if col in df.columns and df[col].isnull().any():
            moda = df[col].mode()
            if not moda.empty:
                df[col] = df[col].fillna(moda[0])
                logger.info("Imputado '%s' con moda='%s'", col, moda[0])
    return df



# =============================================================================
# NODE 1 — LIMPIAR EMPLEADOS
# =============================================================================


def limpiar_empleados(empleados: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = empleados.copy()
    logger.info("=== LIMPIEZA EMPLEADOS | Shape inicial: %s ===", df.shape)

    text_cols = ["nombre", "departamento", "cargo", "tipo_contrato", "jornada"]
    for col in text_cols:
        if col in df.columns:
            df[col] = _normalize_text(df[col])

    df["salario"] = pd.to_numeric(_remove_special_chars(df["salario"]), errors="coerce")
    df["fecha_ingreso"] = _parse_dates_mixed(df["fecha_ingreso"])
    df = _apply_outliers(df, params["outlier_columns"]["empleados"], params)
    df = _impute_nulls(
        df,
        median_cols=params["impute_median"]["empleados"],
        mode_cols=params["impute_mode"]["empleados"],
    )

    dupes = df.duplicated(subset=["id_empleado"], keep="first").sum()
    df = df.drop_duplicates(subset=["id_empleado"], keep="first")
    logger.info("Duplicados eliminados en empleados: %d", dupes)

    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").round(0).astype("Int64")
    df["salario"] = df["salario"].astype(float)

    df["antiguedad_anios"] = (
        (pd.Timestamp.today() - df["fecha_ingreso"]).dt.days / 365.25
    ).round(1)

    logger.info("=== LIMPIEZA EMPLEADOS | Shape final: %s ===", df.shape)
    logger.info("Nulos restantes:\n%s", df.isnull().sum()[df.isnull().sum() > 0])
    df = _limpiar_encoding(df)
    return df



# =============================================================================
# NODE 2 — LIMPIAR AUSENCIAS
# =============================================================================


def limpiar_ausencias(ausencias: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = ausencias.copy()
    logger.info("=== LIMPIEZA AUSENCIAS | Shape inicial: %s ===", df.shape)

    df["tipo_ausencia"] = _normalize_text(df["tipo_ausencia"])
    df["justificada"] = df["justificada"].astype(str).str.strip().replace("Nan", np.nan)

    for col in ["fecha_inicio", "fecha_fin"]:
        df[col] = _parse_dates_mixed(df[col])

    df = _apply_outliers(df, params["outlier_columns"]["ausencias"], params)
    df = _impute_nulls(
        df,
        median_cols=params["impute_median"]["ausencias"],
        mode_cols=params["impute_mode"]["ausencias"],
    )

    dupes = df.duplicated(subset=["id_ausencia"], keep="first").sum()
    df = df.drop_duplicates(subset=["id_ausencia"], keep="first")
    logger.info("Duplicados eliminados en ausencias: %d", dupes)

    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").round(0).astype("Int64")
    df["id_ausencia"] = pd.to_numeric(df["id_ausencia"], errors="coerce").round(0).astype("Int64")
    df["dias"] = df["dias"].astype(float)

    logger.info("=== LIMPIEZA AUSENCIAS | Shape final: %s ===", df.shape)
    df = _limpiar_encoding(df)
    return df



# =============================================================================
# NODE 3 — LIMPIAR CAPACITACIONES
# =============================================================================


def limpiar_capacitaciones(capacitaciones: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = capacitaciones.copy()
    logger.info("=== LIMPIEZA CAPACITACIONES | Shape inicial: %s ===", df.shape)

    df["nombre_curso"] = _normalize_text(df["nombre_curso"])
    df["estado"] = _normalize_text(df["estado"])

    for col in ["fecha_inicio", "fecha_fin"]:
        df[col] = _parse_dates_mixed(df[col])

    df = _apply_outliers(df, params["outlier_columns"]["capacitaciones"], params)

    nota_max = params.get("puntaje_max_valido", 7.0)
    nota_min = params.get("puntaje_min_valido", 1.0)
    df["nota_final"] = df["nota_final"].where(
        df["nota_final"].between(nota_min, nota_max), other=np.nan
    )

    df = _impute_nulls(
        df,
        median_cols=params["impute_median"]["capacitaciones"],
        mode_cols=params["impute_mode"]["capacitaciones"],
    )

    dupes = df.duplicated(subset=["id_capacitacion"], keep="first").sum()
    df = df.drop_duplicates(subset=["id_capacitacion"], keep="first")
    logger.info("Duplicados eliminados en capacitaciones: %d", dupes)

    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").round(0).astype("Int64")
    df["id_capacitacion"] = pd.to_numeric(df["id_capacitacion"], errors="coerce").round(0).astype("Int64")

    logger.info("=== LIMPIEZA CAPACITACIONES | Shape final: %s ===", df.shape)
    df = _limpiar_encoding(df)
    return df



# =============================================================================
# NODE 4 — LIMPIAR EVALUACIONES
# =============================================================================


def limpiar_evaluaciones(evaluaciones: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = evaluaciones.copy()
    logger.info("=== LIMPIEZA EVALUACIONES | Shape inicial: %s ===", df.shape)

    score_cols = ["puntaje_desempeno", "competencias_tecnicas", "competencias_blandas"]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    puntaje_max = params.get("puntaje_max_valido", 7.0)
    puntaje_min = params.get("puntaje_min_valido", 1.0)
    for col in score_cols:
        invalidos = (~df[col].between(puntaje_min, puntaje_max) & df[col].notna()).sum()
        df[col] = df[col].where(df[col].between(puntaje_min, puntaje_max), other=np.nan)
        if invalidos:
            logger.info("Puntajes inválidos en '%s': %d → convertidos a NaN", col, invalidos)

    df = _apply_outliers(df, params["outlier_columns"]["evaluaciones"], params)
    df = _impute_nulls(
        df,
        median_cols=params["impute_median"]["evaluaciones"],
        mode_cols=params["impute_mode"]["evaluaciones"],
    )

    dupes = df.duplicated(subset=["id_evaluacion"], keep="first").sum()
    df = df.drop_duplicates(subset=["id_evaluacion"], keep="first")
    logger.info("Duplicados eliminados en evaluaciones: %d", dupes)

    df["id_empleado"] = pd.to_numeric(df["id_empleado"], errors="coerce").round(0).astype("Int64")
    df["id_evaluacion"] = pd.to_numeric(df["id_evaluacion"], errors="coerce").round(0).astype("Int64")

    logger.info("=== LIMPIEZA EVALUACIONES | Shape final: %s ===", df.shape)
    df = _limpiar_encoding(df)
    return df



# =============================================================================
# NODE 5 — REPORTE DE DIAGNÓSTICO (antes de limpiar)
# =============================================================================


def generar_reporte_diagnostico(
    empleados: pd.DataFrame,
    ausencias: pd.DataFrame,
    capacitaciones: pd.DataFrame,
    evaluaciones: pd.DataFrame,
) -> pd.DataFrame:
    registros = []
    for name, df in [
        ("empleados", empleados),
        ("ausencias", ausencias),
        ("capacitaciones", capacitaciones),
        ("evaluaciones", evaluaciones),
    ]:
        for col in df.columns:
            nulos = df[col].isnull().sum()
            pct_nulos = round(100 * nulos / len(df), 2)
            duplicados = df.duplicated().sum()
            registros.append({
                "dataset": name,
                "columna": col,
                "dtype": str(df[col].dtype),
                "nulos": int(nulos),
                "pct_nulos": pct_nulos,
                "duplicados_fila": int(duplicados),
                "n_filas": len(df),
                "n_unicos": int(df[col].nunique()),
            })

    reporte = pd.DataFrame(registros)
    reporte = _limpiar_encoding(reporte)
    logger.info("Reporte de diagnóstico generado: %d columnas analizadas", len(reporte))
    return reporte
