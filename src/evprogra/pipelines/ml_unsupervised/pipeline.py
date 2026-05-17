"""
pipeline.py — ml_unsupervised
================================
1 nodo principal: aplicar_kmeans
    → KMeans + método del codo + PCA 2D
    → Output: reporte_clustering (catalog → data/08_reporting/)
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aplicar_kmeans


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de aprendizaje no supervisado."""
    return pipeline(
        [
            node(
                func=aplicar_kmeans,
                inputs=[
                    "rrhh_encoded",
                    "params:unsupervised.kmeans.n_clusters",
                    "params:unsupervised.kmeans.n_init",
                    "params:unsupervised.kmeans.max_iter",
                    "params:unsupervised.kmeans.k_range",
                    "params:unsupervised.random_state",
                ],
                outputs="reporte_clustering",
                name="node_kmeans_pca",
            ),
        ]
    )
