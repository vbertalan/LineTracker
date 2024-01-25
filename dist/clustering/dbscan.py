import numpy as np
from .types import ClusteringAlgorithmOutput
import sklearn.cluster as cl

def clustering_dbscan(
    combined_matrix: np.ndarray,
    *,
    epsilon: float,
    **kwargs,
) -> ClusteringAlgorithmOutput:
    """Execute a clustering algorithm using the dbscan algorithm with the epsilon provided. Can benefit from multithreading
    
    # Arguments
    - combined_matrix: np.ndarray, (n_lines, n_lines) assumes without any nan or inf
    
    # Returns
    - ClusteringAlgorithmOutput, see documentation of ClusteringAlgorithmOutput. The clusters of each line is especially in the clustering attribute
    """
    clusterer = cl.DBSCAN(
        eps=epsilon, min_samples=2, metric="precomputed", algorithm="auto", n_jobs=-1
    )
    clusterer.fit(combined_matrix)
    labels = {i: v for i, v in enumerate(clusterer.labels_)}
    return {"type": "dbscan", "clustering": labels, "hyperparameters": {}}
