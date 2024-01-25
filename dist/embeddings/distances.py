import numpy as np
import sklearn.metrics as skMetrics

from typing import *#type: ignore

EmbeddingDistanceType = Literal["cosine", "euclidean"]


def normalized_cosine_distance(data: np.ndarray) -> np.ndarray:
    """Normalized pariwise cosine distance matrix: cosine distance is between 0 and 2 we normalize it between 0 and 1

    # Arguments
    - data: np.ndarray, input embeddings to compute the pairwise distance from

    # Returns
    - np.ndarray, the pairwise cosine distance between each pair of embeddings (data.shape[0],data.shape[0])
    """
    distance = skMetrics.pairwise_distances(data, metric="cosine")
    return distance / 2.0


def normalized_euclidean_distance(data: np.ndarray) -> np.ndarray:
    """Normalized pariwise euclidean distance matrix: euclidean distance does not have bounds  by itself so  we normalize it between 0 and 1 using min max of the distance

    # Arguments
    - data: np.ndarray, input embeddings to compute the pairwise distance from

    # Returns
    - np.ndarray, the pairwise euclidean distance between each pair of embeddings (data.shape[0],data.shape[0])
    """

    distance = skMetrics.pairwise_distances(data, metric="euclidean")
    return (distance - np.min(distance)) / (np.max(distance) - np.min(distance))


def get_embedding_distance_fn(
    embedding_distance: EmbeddingDistanceType,
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a function that can compute the normalized distance between all pairs of embeddings

    # Arguments
    - embedding_distance: EmbeddingDistanceType

    # Returns
    - Callable[[np.ndarray], np.ndarray], a function that takes the embeddings as input and returns the pairwise distance between each pairs of lines (n_lines,n_lines)
    """
    if embedding_distance == "cosine":
        return normalized_cosine_distance
    elif embedding_distance == "euclidean":
        return normalized_euclidean_distance
    else:
        raise ValueError(
            f"Expecting embedding_distance to be among {','.join(get_args(EmbeddingDistanceType))}"
        )
