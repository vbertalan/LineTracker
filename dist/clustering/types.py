from typing import *#type: ignore

class ClusteringAlgorithmOutput(TypedDict):
    """
    - type: str, identifier for the clustering algorithm used
    - clustering: Dict[int, int], mapping of each line to the cluster
    - hyperparameters: Dict[str, Any], hyperparameters used {"hyperparam1": value1, ...}
    """
    type: str
    clustering: Dict[int, int]
    hyperparameters: Dict[str, Any]