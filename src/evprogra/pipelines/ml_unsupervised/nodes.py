"""
nodes.py — ml_unsupervised pipeline
=====================================
Aprendizaje no supervisado:
  1. KMeans sobre rrhh_encoded (segmentación de empleados)
  2. PCA a 2 componentes para visualización
  3. Método del codo para seleccionar k óptimo
  4. Reporte de clustering: silhouette score, inertia, asignación

Inputs (desde catalog):
    rrhh_encoded

Outputs (hacia catalog):
    reporte_clustering
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _seleccionar_features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna solo columnas numéricas sin NaN."""
    X = df.select_dtypes(include=[np.number]).dropna(axis=1)
    return X


def aplicar_kmeans(
    rrhh_encoded: pd.DataFrame,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    k_range: list,
    random_state: int,
) -> pd.DataFrame:
    """
    Aplica KMeans al dataset integrado y evalúa k óptimo con el método del codo.

    Proceso:
        1. Escala features numéricas con StandardScaler.
        2. Calcula inertia y silhouette score para cada k en k_range.
        3. Entrena el modelo final con n_clusters.
        4. Genera reporte con asignación de clusters y métricas.

    Args:
        rrhh_encoded: Dataset final codificado y normalizado.
        n_clusters: Número de clusters para el modelo final.
        n_init: Número de inicializaciones aleatorias de KMeans.
        max_iter: Máximo de iteraciones por inicialización.
        k_range: Lista de valores k a evaluar para el método del codo.
        random_state: Semilla para reproducibilidad.

    Returns:
        reporte_clustering: DataFrame con columnas cluster, pca_x, pca_y,
                            inertia_k_optimo, silhouette_k_optimo y métricas
                            de evaluación por k.
    """
    X = _seleccionar_features_numericas(rrhh_encoded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Método del codo ──────────────────────────────────────────────────────
    codo_resultados = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_) if k > 1 else None
        codo_resultados.append({
            "k":          k,
            "inertia":    round(km.inertia_, 4),
            "silhouette": round(sil, 4) if sil is not None else None,
        })
        logger.info("k=%d | inertia=%.2f | silhouette=%s", k, km.inertia_,
                    f"{sil:.4f}" if sil else "N/A")

    codo_df = pd.DataFrame(codo_resultados)

    # ── Modelo final ─────────────────────────────────────────────────────────
    km_final = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    km_final.fit(X_scaled)
    labels      = km_final.labels_
    inertia_fin = km_final.inertia_
    sil_fin     = silhouette_score(X_scaled, labels)

    logger.info(
        "Modelo final | k=%d | inertia=%.4f | silhouette=%.4f",
        n_clusters, inertia_fin, sil_fin,
    )

    # Distribución de clusters
    unique, counts = np.unique(labels, return_counts=True)
    for cl, cnt in zip(unique, counts):
        logger.info("  Cluster %d: %d empleados (%.1f%%)", cl, cnt, 100*cnt/len(labels))

    # ── PCA 2D ───────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    var_explicada = pca.explained_variance_ratio_
    logger.info(
        "PCA | varianza explicada: PC1=%.3f PC2=%.3f (total=%.3f)",
        var_explicada[0], var_explicada[1], var_explicada.sum()
    )

    # ── Reporte final ─────────────────────────────────────────────────────────
    reporte = pd.DataFrame({
        "id_empleado":      rrhh_encoded["id_empleado"].values if "id_empleado" in rrhh_encoded.columns else range(len(rrhh_encoded)),
        "cluster":          labels,
        "pca_componente_1": X_pca[:, 0].round(4),
        "pca_componente_2": X_pca[:, 1].round(4),
    })

    # Agregar métricas globales como columnas informativas
    reporte["n_clusters_final"]    = n_clusters
    reporte["inertia_final"]       = round(inertia_fin, 4)
    reporte["silhouette_final"]    = round(sil_fin, 4)
    reporte["pca_varianza_pc1"]    = round(float(var_explicada[0]), 4)
    reporte["pca_varianza_pc2"]    = round(float(var_explicada[1]), 4)
    reporte["pca_varianza_total"]  = round(float(var_explicada.sum()), 4)

    # Agregar tabla del codo como filas de resumen separadas por k
    reporte_codo = codo_df.rename(columns={
        "k":          "cluster",
        "inertia":    "inertia_final",
        "silhouette": "silhouette_final",
    })
    reporte_codo["tipo"] = "codo"
    reporte["tipo"]      = "empleado"

    reporte_final = pd.concat([reporte, reporte_codo], ignore_index=True)

    return reporte_final
