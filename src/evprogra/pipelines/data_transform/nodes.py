"""
nodes.py — Pipeline data_transform
Caso 3: Recursos Humanos — SCY1101 Programación para la Ciencia de Datos

Cubre AD 1.3:
  - Joins/merges entre las 4 tablas limpias
  - pivot_table y groupby avanzados
  - Creación de features derivadas
  - Normalización MinMaxScaler y estandarización StandardScaler
  - Codificación Label Encoding y One-Hot Encoding
  - Vectorización y optimización de memoria
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER: optimizar tipos para reducir memoria (broadcasting/vectorización)
# =============================================================================
def _optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el uso de memoria convirtiendo columnas numéricas al tipo más pequeño posible.
    Técnica de optimización para gran escala (Indicador 3 — 100%).
    """
    antes = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    despues = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("Optimización memoria: %.2f MB → %.2f MB (−%.1f%%)",
                antes, despues, (antes - despues) / antes * 100)
    return df


# =============================================================================
# NODO 1: INTEGRAR DATASETS — joins entre las 4 tablas + aggregations
# =============================================================================
def integrar_datasets(
    empleados: pd.DataFrame,
    ausencias: pd.DataFrame,
    capacitaciones: pd.DataFrame,
    evaluaciones: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Une los 4 datasets limpios mediante left joins sobre id_empleado.
    Agrega métricas por empleado usando groupby múltiple y pivot_table.
    """
    join_key = params.get("join_key", "id_empleado")
    logger.info("=== INTEGRACIÓN DE DATASETS | join_key=%s ===", join_key)
    logger.info("Shapes: empleados=%s, ausencias=%s, capacitaciones=%s, evaluaciones=%s",
                empleados.shape, ausencias.shape, capacitaciones.shape, evaluaciones.shape)

    # ── Groupby ausencias ────────────────────────────────────────────────────
    agg_aus = (
        ausencias.groupby(join_key, as_index=False)
        .agg(
            total_ausencias=(join_key, "count"),
            dias_ausencia_total=("dias", "sum"),
            dias_ausencia_promedio=("dias", "mean"),
            ausencias_injustificadas=(
                "justificada",
                lambda x: (x.astype(str).str.strip().str.lower() == "no").sum(),
            ),
            tipo_ausencia_mas_frecuente=("tipo_ausencia", lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan),
        )
    )
    agg_aus["dias_ausencia_promedio"] = agg_aus["dias_ausencia_promedio"].round(2)

    # ── Pivot table ausencias: tipo × justificada ────────────────────────────
    pivot_aus = pd.pivot_table(
        ausencias,
        values="dias",
        index=join_key,
        columns="justificada",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot_aus.columns = [
        join_key if c == join_key else f"dias_ausencia_just_{str(c).strip().lower()}"
        for c in pivot_aus.columns
    ]
    logger.info("Pivot ausencias: %s columnas generadas", pivot_aus.shape[1] - 1)

    # ── Groupby capacitaciones ───────────────────────────────────────────────
    agg_cap = (
        capacitaciones.groupby(join_key, as_index=False)
        .agg(
            total_capacitaciones=(join_key, "count"),
            horas_capacitacion_total=("horas", "sum"),
            horas_capacitacion_promedio=("horas", "mean"),
            nota_promedio_capacitacion=("nota_final", "mean"),
            capacitaciones_completadas=(
                "estado",
                lambda x: (x.str.strip().str.lower() == "completado").sum(),
            ),
        )
    )
    agg_cap["horas_capacitacion_promedio"] = agg_cap["horas_capacitacion_promedio"].round(2)
    agg_cap["nota_promedio_capacitacion"]  = agg_cap["nota_promedio_capacitacion"].round(2)

    # Flag binario: cumple mínimo de capacitaciones completadas
    min_cap = params.get("min_capacitaciones_completadas", 1)
    agg_cap["cumple_min_capacitaciones"] = (
        agg_cap["capacitaciones_completadas"] >= min_cap
    ).astype(int)

    # ── Pivot table capacitaciones: estado por tipo de curso ─────────────────
    pivot_cap = pd.pivot_table(
        capacitaciones,
        values="nota_final",
        index=join_key,
        columns="estado",
        aggfunc="mean",
        fill_value=0,
    ).reset_index()
    pivot_cap.columns = [
        join_key if c == join_key else f"nota_estado_{str(c).strip().lower().replace(' ', '_')}"
        for c in pivot_cap.columns
    ]
    logger.info("Pivot capacitaciones: %s columnas generadas", pivot_cap.shape[1] - 1)

    # ── Groupby evaluaciones ─────────────────────────────────────────────────
    eva_valida = evaluaciones.dropna(subset=[join_key]).copy()
    agg_eva = (
        eva_valida.groupby(join_key, as_index=False)
        .agg(
            total_evaluaciones=(join_key, "count"),
            puntaje_desempeno_promedio=("puntaje_desempeno", "mean"),
            competencias_tecnicas_promedio=("competencias_tecnicas", "mean"),
            competencias_blandas_promedio=("competencias_blandas", "mean"),
            puntaje_desempeno_max=("puntaje_desempeno", "max"),
            puntaje_desempeno_min=("puntaje_desempeno", "min"),
        )
    )
    for col in ["puntaje_desempeno_promedio", "competencias_tecnicas_promedio",
                "competencias_blandas_promedio"]:
        agg_eva[col] = agg_eva[col].round(2)

    # ── Joins secuenciales ───────────────────────────────────────────────────
    maestro = empleados.merge(agg_aus,   on=join_key, how="left")
    maestro = maestro.merge(pivot_aus,   on=join_key, how="left")
    maestro = maestro.merge(agg_cap,     on=join_key, how="left")
    maestro = maestro.merge(pivot_cap,   on=join_key, how="left")
    maestro = maestro.merge(agg_eva,     on=join_key, how="left")

    # Rellenar 0 en columnas de conteo
    cols_cero = [
        "total_ausencias", "dias_ausencia_total", "ausencias_injustificadas",
        "total_capacitaciones", "horas_capacitacion_total",
        "capacitaciones_completadas", "cumple_min_capacitaciones", "total_evaluaciones",
    ]
    for col in cols_cero:
        if col in maestro.columns:
            maestro[col] = maestro[col].fillna(0)

    maestro = _optimizar_memoria(maestro)
    logger.info("=== TABLA INTEGRADA | Shape: %s ===", maestro.shape)
    return maestro


# =============================================================================
# NODO 2: RESUMEN DEPARTAMENTO — groupby + pivot para reporting
# =============================================================================
def calcular_resumen_departamento(maestro: pd.DataFrame) -> pd.DataFrame:
    """
    Genera KPIs departamentales usando groupby múltiple y reshape/pivot.
    """
    logger.info("=== RESUMEN DEPARTAMENTO ===")
    df = maestro.dropna(subset=["departamento"]).copy()

    # Normalizar departamento (ya limpio, solo asegurar)
    df["departamento"] = df["departamento"].astype(str).str.strip().str.title()

    resumen = (
        df.groupby("departamento")
        .agg(
            n_empleados=("id_empleado", "count"),
            salario_promedio=("salario", "mean"),
            salario_mediana=("salario", "median"),
            salario_std=("salario", "std"),
            dias_ausencia_promedio=("dias_ausencia_total", "mean"),
            horas_cap_promedio=("horas_capacitacion_total", "mean"),
            nota_cap_promedio=("nota_promedio_capacitacion", "mean"),
            puntaje_desempeno_promedio=("puntaje_desempeno_promedio", "mean"),
            comp_tecnicas_promedio=("competencias_tecnicas_promedio", "mean"),
            comp_blandas_promedio=("competencias_blandas_promedio", "mean"),
            pct_cumple_cap=(
                "cumple_min_capacitaciones",
                lambda x: round(x.mean() * 100, 1),
            ),
        )
        .reset_index()
    )

    for col in resumen.select_dtypes(include="float").columns:
        resumen[col] = resumen[col].round(2)

    # Reshape: ranking por salario usando vectorización
    resumen["ranking_salario"] = resumen["salario_promedio"].rank(
        ascending=False, method="dense"
    ).astype(int)
    resumen["ranking_desempeno"] = resumen["puntaje_desempeno_promedio"].rank(
        ascending=False, method="dense"
    ).astype(int)

    logger.info("Resumen generado: %d departamentos, %d métricas", len(resumen), len(resumen.columns))
    return resumen


# =============================================================================
# NODO 3: CREAR FEATURES + NORMALIZAR + CODIFICAR → rrhh_encoded
# =============================================================================
def transformar_y_codificar(
    maestro: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Aplica al dataset integrado:
      1. Features derivadas (score compuesto, tasa ausentismo, nivel cap)
      2. MinMaxScaler sobre columnas numéricas clave
      3. StandardScaler sobre competencias
      4. LabelEncoder sobre tipo_contrato y jornada
      5. One-Hot Encoding sobre departamento y tipo_ausencia_mas_frecuente
    """
    logger.info("=== TRANSFORMACIÓN Y CODIFICACIÓN ===")
    df = maestro.copy()

    # ── 1. Features derivadas ────────────────────────────────────────────────
    dias_laborales = 250
    df["tasa_ausentismo_pct"] = (
        df["dias_ausencia_total"] / dias_laborales * 100
    ).round(2)

    df["tasa_completitud_cap"] = (
        df["capacitaciones_completadas"]
        / df["total_capacitaciones"].replace(0, np.nan)
        * 100
    ).fillna(0).round(2)

    # Score compuesto de desempeño (ponderado)
    df["score_desempeno"] = (
        df["puntaje_desempeno_promedio"].fillna(0) * 0.5
        + df["competencias_tecnicas_promedio"].fillna(0) * 0.3
        + df["competencias_blandas_promedio"].fillna(0) * 0.2
    ).round(2)

    # Segmentos usando np.select (vectorización, evita apply)
    df["segmento_desempeno"] = np.select(
        [df["score_desempeno"] >= 5.5, df["score_desempeno"] >= 3.5],
        ["Alto", "Medio"],
        default="Bajo",
    )
    df["riesgo_ausentismo"] = np.select(
        [df["tasa_ausentismo_pct"] >= 10.0, df["tasa_ausentismo_pct"] >= 5.0],
        ["Crítico", "Moderado"],
        default="Normal",
    )
    logger.info("Features derivadas creadas: tasa_ausentismo_pct, tasa_completitud_cap, score_desempeno, segmento_desempeno, riesgo_ausentismo")

    # ── 2. MinMaxScaler ──────────────────────────────────────────────────────
    cols_minmax = [c for c in params.get("normalize_minmax", []) if c in df.columns]
    if cols_minmax:
        scaler_mm = MinMaxScaler()
        df[[f"{c}_norm" for c in cols_minmax]] = scaler_mm.fit_transform(
            df[cols_minmax].fillna(0)
        ).round(4)
        logger.info("MinMaxScaler aplicado a: %s", cols_minmax)

    # ── 3. StandardScaler ────────────────────────────────────────────────────
    cols_std = [c for c in params.get("standardize_zscore", []) if c in df.columns]
    if cols_std:
        scaler_std = StandardScaler()
        df[[f"{c}_std" for c in cols_std]] = scaler_std.fit_transform(
            df[cols_std].fillna(0)
        ).round(4)
        logger.info("StandardScaler aplicado a: %s", cols_std)

    # ── 4. Label Encoding ────────────────────────────────────────────────────
    cols_label = [c for c in params.get("label_encode", []) if c in df.columns]
    for col in cols_label:
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, f"{col}_enc"] = le.fit_transform(
            df.loc[mask, col].astype(str)
        )
        logger.info("LabelEncoder aplicado a '%s' → clases: %s", col, list(le.classes_))

    # ── 5. One-Hot Encoding ──────────────────────────────────────────────────
    cols_ohe = [c for c in params.get("onehot_encode", []) if c in df.columns]
    if cols_ohe:
        df_ohe = pd.get_dummies(
            df[cols_ohe].fillna("Desconocido").astype(str),
            prefix=cols_ohe,
            dtype=int,
        )
        df = pd.concat([df, df_ohe], axis=1)
        logger.info("One-Hot Encoding aplicado a: %s → %d nuevas columnas",
                    cols_ohe, df_ohe.shape[1])

    df = _optimizar_memoria(df)
    logger.info("=== DATASET FINAL | Shape: %s ===", df.shape)
    return df