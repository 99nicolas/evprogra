"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline

# ── EV1 — Pipelines de datos ─────────────────────────────────────────────────
from evprogra.pipelines.data_ingestion  import create_pipeline as create_data_ingestion
from evprogra.pipelines.data_cleaning   import create_pipeline as create_data_cleaning
from evprogra.pipelines.data_transform  import create_pipeline as create_data_transform
from evprogra.pipelines.data_validation import create_pipeline as create_data_validation

# ── EV2 — Pipelines de Machine Learning ──────────────────────────────────────
from evprogra.pipelines.data_preprocessing import create_pipeline as create_data_preprocessing
from evprogra.pipelines.ml_classification  import create_pipeline as create_ml_classification
from evprogra.pipelines.ml_regression      import create_pipeline as create_ml_regression
from evprogra.pipelines.ml_unsupervised    import create_pipeline as create_ml_unsupervised
from evprogra.pipelines.ml_optimization    import create_pipeline as create_ml_optimization


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    EV1 pipelines (datos):
        data_ingestion   → exploración y diagnóstico inicial
        data_cleaning    → limpieza de los 4 datasets
        data_transform   → integración y transformaciones avanzadas
        data_validation  → verificación de integridad

    EV2 pipelines (machine learning):
        data_preprocessing → split train/test para clf y reg
        ml_classification  → RF, LR, GBM sobre segmento_desempeno
        ml_regression      → RF, Ridge, GBM sobre score_desempeno
        ml_unsupervised    → KMeans + PCA
        ml_optimization    → GridSearchCV + RandomizedSearchCV

    Shortcuts:
        ev1          → flujo completo EV1
        ev2          → flujo completo EV2 (requiere rrhh_encoded en 03_primary)
        __default__  → ev1 + ev2
    """

    # ── Instanciar EV1 ────────────────────────────────────────────────────────
    data_ingestion  = create_data_ingestion()
    data_cleaning   = create_data_cleaning()
    data_transform  = create_data_transform()
    data_validation = create_data_validation()

    # ── Instanciar EV2 ────────────────────────────────────────────────────────
    data_preprocessing = create_data_preprocessing()
    ml_classification  = create_ml_classification()
    ml_regression      = create_ml_regression()
    ml_unsupervised    = create_ml_unsupervised()
    ml_optimization    = create_ml_optimization()

    # ── Flujos combinados ─────────────────────────────────────────────────────
    pipeline_ev1 = data_cleaning + data_transform + data_validation
    pipeline_ev2 = (
        data_preprocessing
        + ml_classification
        + ml_regression
        + ml_unsupervised
        + ml_optimization
    )

    return {
        # ── EV1 — individuales ────────────────────────────────────────────────
        "data_ingestion":   data_ingestion,
        "data_cleaning":    data_cleaning,
        "data_transform":   data_transform,
        "data_validation":  data_validation,

        # ── EV2 — individuales ────────────────────────────────────────────────
        "data_preprocessing": data_preprocessing,
        "ml_classification":  ml_classification,
        "ml_regression":      ml_regression,
        "ml_unsupervised":    ml_unsupervised,
        "ml_optimization":    ml_optimization,

        # ── Flujos combinados ─────────────────────────────────────────────────
        "ev1":         pipeline_ev1,
        "ev2":         pipeline_ev2,
        "__default__": pipeline_ev1 + pipeline_ev2,
    }
