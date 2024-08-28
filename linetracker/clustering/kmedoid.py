"""Contains all function to do the kmedoid algorithm. Note that the c++ kmedoid should have been compiled before using these functions (the path to the c++ library is an argument of get_clustering_kmedoid)"""

import contextlib
import numpy as np
import gc
import ctypes
import sklearn.metrics as skMetrics

from .types import ClusteringAlgorithmOutput
from pathlib import Path
from typing import *  # type: ignore
import time


class ClusteringAlgorithmOutputKMedoid(ClusteringAlgorithmOutput):
    """Output of the clustering with kmedoid
    
    - type: str, the name of the clustering (will be kmedoid here)
    - clustering: Dict[int, int], the clusters for each line {line_0: cluster_0, ...}
    - hyperparameters: Dict[str, Any], {"name_parameter_1": value, ...} and here will be {"number_of_clusters": ...}
    - score: float, the score found associated with the number of cluster (silhouette score)
    """
    type: str
    clustering: Dict[int, int]
    hyperparameters: Dict[str, Any]
    score: float


def comply_with_num_of_clusters(
    distances: np.ndarray,
    clusters: np.ndarray,
    must_link: List[Tuple[int, int]],
    target_n_clusters: int,
) -> np.ndarray:
    """As the kmedoid algorithm can provide less clusters than asked, this method makes the missing clusters by using the most distant points as single clusters. The most distant points are the points with the biggest cumulative distance. We exclude points that are in must_link constraints and points that are already single point clusters

    # Arguments
    - distances: np.ndarray, (n_lines,n_lines) the distance matrix between all of the points
    - clusters: np.ndarray, (n_lines,) the clusters chosen for each line that can be less than target_n_clusters
    - must_link: List[Tuple[int, int]], the points that must be in the same cluster, to avoid them to be in different clusters
    - target_n_clusters: int, the required number of clusters

    # Returns
    - np.ndarray, (n_lines,) the neww clusters for each line

    # Example usage

    ```python
    >>> n_lines = 10
    >>> target_n_clusters = 3
    >>> distances = np.random.rand(n_lines, n_lines)
    >>> distances = (distances+distances.T)/2.
    >>> np.fill_diagonal(distances, 0)
    >>> clusters = np.random.randint(0,2,(n_lines,))
    >>> must_link = []
    >>> comply_with_num_of_clusters(distances, clusters, must_link, target_n_clusters)
    ```
    """
    clusters = np.array(clusters)
    # 1. Setup cumulated distances (dont change and will be consumed progressively), unique clusters and next_cluster index
    # we will chose point with the biggest cumulative distance to all other points in priority
    # cum_distance(point_i) = sum(distance(i,j) for j in range(n_points))
    cum_distances = np.sum(distances, axis=1)
    cum_distances = [(i, cum_distances[i]) for i in range(len(distances))]
    # extract the points with the biggest cumulative distance
    cum_distances.sort(key=lambda x: x[1], reverse=False)
    # we initialize the unique clusters found and the next cluster name
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    # this next cluster is taken bigger as the biggest cluster index
    # an other incorrect solution would have been to use the point index as a unique cluster but it could have been used by the point and other points
    next_cluster = max(unique_clusters) + 1
    # 2. we loop up until we have the required number of clusters
    while len(unique_clusters) != target_n_clusters:
        # 2.1 build the potential points
        potential_points = set(range(len(clusters)))
        # remove points that are alreay single cluster
        for p, c in zip(unique_clusters, counts):
            if c == 1:
                potential_points.difference([p])
        # remove the must link, simplified reasoning as we want to create new single point clusters: impossible if a point is in one must link constraint
        for p1, p2 in must_link:
            potential_points.difference([p1, p2])
        potential_points = sorted(potential_points)
        # 2.2 get the most distant clusters that is in the allowed points as single point cluster
        i = None
        while i is None or i not in potential_points:
            i, _ = cum_distances.pop(-1)
        # 2.3 If we reached the end of potential points in cumulated distance, it means that it is not possible to comply with the requirements either because there are too many clusters required (target_n_clusters>n_points) or because the must_link constraints limit the possibility
        if len(cum_distances) == 0:
            raise Exception(
                f"Cannot comply with the number of clusters due to constraints {must_link=} for {target_n_clusters=}: wwe have max {len(unique_clusters)=} with {unique_clusters=} and {clusters=} and potential clusters {[e for e,c in zip(unique_clusters,counts) if c > 1]} and {len(clusters)=}"
            )
        # 2.4 we update the new cluster and update the linked variables (next_cluster and unique_clusters, counts)
        clusters[i] = next_cluster
        next_cluster += 1
        unique_clusters, counts = np.unique(clusters, return_counts=True)
    # 3. When we reached the desired number of clusters, we can return the clusters
    return clusters


def clustering_kmedoids(
    combined_matrix: np.ndarray,
    *,
    number_of_clusters: int,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    iteration_max: Optional[int] = None,
    seed: int = 0,
    library_path: Union[str, Path] = "./clustering_lib.so",
    time_limit: float = 0.2,
    **kwargs,
) -> Tuple[Dict[int, int], int, float]:
    """Function to call the c++ already compiled kmedoid.so library. Starting from a random clustering it iteratively improves it ensuring that points in must_link pars are always linked together and points in cannot_link are never linked together. **It ensures that the number_of_clusters asked is respected** by using comply_with_num_of_clusters
    
    # Arguments
    - combined_matrix: np.ndarray, (n_lines, n_lines) assumes without any nan or inf
    - *, to ensure that the following arguments are always provided by argument
    - number_of_clusters: Optional[int] = None, required number of clusters
    - must_link: Optional[List[Tuple[int, int]]] = None, if None [], each tuple represents the indexes of two points that must be linked together : example [(0,1),(2,0)] will ensure that the points at index 0 andd 1 and 2 are always in the same cluster (because 0 and 1 together and 2 and 0 together)
    - cannot_link: Optional[List[Tuple[int, int]]] = None, if None [],  each tuple represents the indexes of two points that must not be linked together : example [(0,1),(2,0)] will ensure that the points at index 0 and 1 are never in the same cluster and that 2 and 0 are never in the same cluster (but here 1 and 2 can be together)
    - iteration_max: Optional[int] = None, represents the number of loops done by the local search algorithm but also is linked to the number of shaking at each step
    - seed: int = 0, the random seed to use for the algorithm (reproductubility not guaranteed at the time)
    - library_path: Union[str, Path] = "./clustering_lib.so", the path to the library to run the kmeoids c++ algorithm
    
    # Returns
    - Dict[int, int], for each line number the cluster associated
    """
    # 1. Prepare arguments, default values...
    library_path = Path(library_path).resolve()
    assert library_path.exists(), f"{library_path.as_posix()=} does not exist"
    library_path = library_path.as_posix() #type: ignore
    if must_link is None:
        must_link = []
    if cannot_link is None:
        cannot_link = []
    if iteration_max is None:
        iteration_max = 10
    if time_limit is None:
        time_limit = 0.2
    if number_of_clusters == 1:
        return {i: 0 for i in range(len(combined_matrix))}
    if number_of_clusters == len(combined_matrix):
        return {i: c for i, c in enumerate(range(len(combined_matrix)))}
    # 2. Prepare the matrix (flatten as easier to transfer to c++)
    n_points, n_dims = combined_matrix.shape
    combined_matrix = combined_matrix.flatten()
    # 3. Load the c++ library
    dummy_cpp_library = ctypes.CDLL(library_path)
    # 4. Prepare the arguments and their associated type for c++ transfer
    combine_matrix_ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
    must_link_array = np.array(must_link).flatten().astype(np.int32)
    must_link_array_ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
    # must_link_array_ptr = must_link_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    n_must_link = len(must_link)
    cannot_link_array = np.array(cannot_link).flatten().astype(np.int32)
    cannot_link_array_ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C")
    n_cannot_link = len(cannot_link)
    # 5. Prepare the function prototype: input arguments types and return type
    dummy_cpp_library.clusterize.argtypes = [
        ctypes.c_int,
        combine_matrix_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        must_link_array_ptr,
        ctypes.c_size_t,
        cannot_link_array_ptr,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_bool,
    ]
    dummy_cpp_library.clusterize.restype = ctypes.POINTER(ctypes.c_double)
    # 6. Run the function
    start_time = time.perf_counter()
    data_ptr = dummy_cpp_library.clusterize(
        seed,
        combined_matrix.astype(np.float64),
        int(n_points),
        int(n_dims),
        int(number_of_clusters),
        must_link_array,
        int(n_must_link),
        cannot_link_array,
        n_cannot_link,
        iteration_max,
        time_limit,
        True,
    )
    duration = time.perf_counter()-start_time
    # 7. Convert and get the clustering result and check that no suspicious cluster is seen
    data_out = np.copy(np.ctypeslib.as_array(data_ptr, shape=(n_points+2,)))
    clusters = data_out[2:]
    num_iter = int(data_out[0])
    final_objective = data_out[1]
    unique_clusters = np.unique(clusters)
    assert (
        len(unique_clusters) <= number_of_clusters
    ), f"Expecting {number_of_clusters=} or less but found {len(unique_clusters)=} with {unique_clusters=} {clusters=}\n{combined_matrix.reshape((n_points, n_dims)).tolist()=}"
    
    # 9. Ensure that we comply with the required number of clusters
    clusters = comply_with_num_of_clusters(
        distances=combined_matrix.reshape((n_points, n_dims)),
        clusters=clusters,
        must_link=must_link,
        target_n_clusters=number_of_clusters,
    )
    clusters = {i: c for i, c in enumerate(clusters)}
    # 8. Free the last pointer not already freed
    try:
        dummy_cpp_library.free_array(data_ptr)
    except Exception:
        pass
    return clusters, num_iter, final_objective, duration


def get_clustering_kmedoid(
    combined_matrix: np.ndarray,
    *,
    must_link: Optional[List[Tuple[int, int]]] = None,
    cannot_link: Optional[List[Tuple[int, int]]] = None,
    iteration_max: Optional[int] = None,
    seed: int = 0,
    library_path: Union[str, Path] = "./clustering_lib.so",
    n_samples: int = 10,
    **kwargs,
) -> List[ClusteringAlgorithmOutputKMedoid]:
    """Wrapper around the clustering_kmedoids function: the task of get_clustering_kmedoid is to automatically choose the best number of clusters based on the silhouette score (best score for choosing number of clusters when we do not know it)
    
    # Arguments
    - combined_matrix: np.ndarray, (n_lines, n_lines) assumes without any nan or inf
    - *, to ensure that the following arguments are always provided by argument
    - must_link: Optional[List[Tuple[int, int]]] = None, if None [], each tuple represents the indexes of two points that must be linked together : example [(0,1),(2,0)] will ensure that the points at index 0 andd 1 and 2 are always in the same cluster (because 0 and 1 together and 2 and 0 together)
    - cannot_link: Optional[List[Tuple[int, int]]] = None, if None [],  each tuple represents the indexes of two points that must not be linked together : example [(0,1),(2,0)] will ensure that the points at index 0 and 1 are never in the same cluster and that 2 and 0 are never in the same cluster (but here 1 and 2 can be together)
    - iteration_max: Optional[int] = None, represents the number of loops done by the local search algorithm but also is linked to the number of shaking at each step
    - seed: int = 0, the random seed to use for the algorithm (reproductubility not guaranteed at the time)
    - library_path: Union[str, Path] = "./clustering_lib.so", the path to the library to run the kmeoids c++ algorithm
    - n_samples: int = 10, the number of clusters tested. Do not guarantee that we will test this many number of clusters: We can comply with this value if it is less than the limit number of clusters tested of (n_points-1) - 2 + 1: limits imposed by the silhouette score that the number of clusters is in the interval [2,n_points-1] see [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). 
    
    # Returns
    - ClusteringAlgorithmOutputKMedoid, see documentation of ClusteringAlgorithmOutputKMedoid. The clusters of each line is especially in the clustering attribute
    """
    # 1. We start to clean the memory of unused object (to avoid memory errors due to the c++ program calls)
    gc.collect()
    # 2. Exclude the simple cases: 0 lines, 1 lines -> 1 cluster, !! 2 lines is chosen as 2 clusters !!
    # We need to exclude the case of 2 lines because the silhouette score can only be computed for 2 -> n-1 clusters with n the number of lines
    if len(combined_matrix) == 0:
        return [{
            "type": "kmedoids",
            "score": 1.0,
            "clustering": {},
            "hyperparameters": {"number_of_clusters": -1},
        }]
    if len(combined_matrix) == 1:
        return [{
            "type": "kmedoids",
            "score": 1.0,
            "clustering": {0: 0},
            "hyperparameters": {"number_of_clusters": 1},
        }]
    if len(combined_matrix) == 2:
        return [{
            "type": "kmedoid",
            "clustering": {0: 0, 1: 1},
            "hyperparameters": {"number_of_clusters": -1},
            "score": 1,
        }]
    # 3. Manages the default values
    if must_link is None:
        must_link = []
    if cannot_link is None:
        cannot_link = []
    if iteration_max is None:
        iteration_max = 10
    # 4. Then we test the number of clusters to try to maximize the silhouette score (score in ClusteringAlgorithmOutputKMedoid dict)
    # best: ClusteringAlgorithmOutputKMedoid = {
    #     "type": "kmedoid",
    #     "clustering": {},
    #     "hyperparameters": {"number_of_clusters": -1},
    #     "score": -float("inf"),
    # }
    # 4.1 We will samples the number of clusters between the maximum number of clusters and the minimum number of clusters:
    # The maximum number of samples will be either n_samples if we have a lot of lines or len(combined_matrix) - 1 - 2 + 1:
    # len(combined_matrix) - 1 is the maximum number of samples supported by silhouette score
    # 2 is the minimum number of samples supported by silhouette score
    # len(combined_matrix) - 1 - 2 + 1 is the number of elements inside this interval
    # so we choose either the maximum number of samples possible if n_samples is bigger than this limit or we choose n_samples if possible
    n_samples = min(n_samples, len(combined_matrix) - 1 - 2 + 1)
    clusters_to_test = set( # set to exclude duplicates due to round/int cast
        np.round(
            np.linspace(2, len(combined_matrix) - 1, n_samples) # comply with silhouette score requirements
        ).astype(int) # have integers
    )
    Lout = []
    for number_of_clusters_tested in clusters_to_test:
        clustering, num_iter, final_objective, duration = clustering_kmedoids(
            combined_matrix,
            must_link=must_link,
            cannot_link=cannot_link,
            iteration_max=iteration_max,
            seed=seed,
            number_of_clusters=number_of_clusters_tested,
            library_path=library_path,
        )
        # 4.2 We check if there the silhouette score
        score = float(
            skMetrics.silhouette_score(
                X=combined_matrix,
                labels=[clustering[i] for i in range(len(combined_matrix))],
                metric="precomputed",
            )
        )
        Lout.append({
                "type": "kmedoid",
                "clustering": clustering,
                "hyperparameters": {"number_of_clusters": number_of_clusters_tested},
                "score": score,
                "num_iter":num_iter, 
                "final_objective":final_objective, 
                "duration":duration,
        })
    # 5. We return the best clustering found according to silhouette score
    return Lout
