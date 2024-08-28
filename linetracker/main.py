"""Main file containing especially execute_full_pipeline to run the full pipeline. Main entry point."""
from typing import *  # type: ignore
import torch
import transformers as trf
import numpy as np
import linetracker.parser.parser as p
import linetracker.clustering.types as c
import linetracker.parser.variables_matrix as e

"""File of code: disclaimer functions comes from the repository https://github.com/AndressaStefany/severityPrediction"""
# tye hints
LlamaTokenizer = Union["trf.LlamaTokenizer", "trf.LlamaTokenizerFast"]
LlamaModel = "trf.LlamaForCausalLM"
PoolingOperationCode = Literal["mean", "sum"]
PoolingFn = Callable[["torch.Tensor"], "torch.Tensor"]
ModelName = Literal["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
DatasetName = Literal["eclipse_72k", "mozilla_200k"]
BugId: int
ParserTypes = Literal["drain"]
EmbedderType = Literal["llama-13b", "tfidf"]
EmbeddingDistanceType = Literal["cosine", "euclidean"]
ClusteringType = Literal["kmedoid", "dbscan"]
# typehint imports


class LogData(TypedDict):
    """
    - event_id: str, Unique string per bug_id
    - text: str, Text of the error
    - line_num: str, Plan id of the log: with log_name constitute the build_log
    """

    event_id: str
    text: str
    line_num: str


class TripletMatrix(TypedDict):
    """
    - variables_matrix: np.ndarray, matrix distances
    - embeddings_matrix: np.ndarray, embeddings distances
    - count_matrix: np.ndarray, count of line distances
    """

    variables_matrix: np.ndarray
    embeddings_matrix: np.ndarray
    count_matrix: np.ndarray


class TripletCoef(TypedDict):
    """
    - coef_variables_matrix: coefficient for the matrix distances
    - coef_embeddings_matrix: coefficient for the embeddings distances
    - coef_count_matrix: coefficient for the count of line distances
    """

    coef_variables_matrix: float
    coef_embeddings_matrix: float
    coef_count_matrix: float




def combine_matrices(
    triplet_matrix: TripletMatrix,
    triplet_coef: TripletCoef,
    precision=np.float32,
    **kwargs,
) -> np.ndarray:
    """Combine triplet matrices into one matrix using precision"""
    combined_matrix = np.zeros(
        triplet_matrix["embeddings_matrix"].shape, dtype=precision
    )
    for name in ["embeddings_matrix", "variables_matrix", "count_matrix"]:
        combined_matrix += triplet_matrix[name] * triplet_coef[f"coef_{name}"]
    return combined_matrix

def execute_full_pipeline(
    logs: List[LogData],
    triplet_coefficient: TripletCoef,
    parser: Callable[[List[LogData]], List[p.ParsedLine]],
    embedder: Callable[[List[str]], Generator[np.ndarray, None, None]],
    embedding_distance_fn: Callable[[List[np.ndarray]], np.ndarray],
    line_distance_fn: Callable[[List[LogData]], np.ndarray],
    clustering_fn: Callable[[np.ndarray], c.ClusteringAlgorithmOutput],
    float_precision: type = np.float32,
) -> c.ClusteringAlgorithmOutput:
    """Cluster logs provided in argument into groups of related log lines
    # Arguments
    - logs: List[LogData], the log lines
    - triplet_coefficient: TripletCoef, the three coefficients to use to ponderate the matrices
    - parser: Callable[[List[LogData]], List[p.ParsedLine]], a function that from the list of logs lines can generate for each line
    - embedder: Callable[[List[str]], Generator[np.ndarray, None, None]], the function that can generate embeddings from logs
    - embedding_distance_fn: Callable[[List[np.ndarray]], np.ndarray], given all embeddings of each log lines of the same log file, generate the normalized (between 0 and 1) distances between all embeddings
    - line_distance_fn: Callable[[List[str]],np.ndarray], a function that can generate a matrix with the distance between each log line
    - clustering_fn:  Callable[[np.ndarray], c.ClusteringAlgorithmOutput], taking the combined matrix with the coefficients provided, clusters the logs
    - float_precision: type = np.float32, the precision to use for all floating point matrices
    """
    # 1. parse the logs
    parsed_logs: List[p.ParsedLine] = parser(logs)
    logs_texts = [e["text"] for e in logs]
    parsed_variables = [e["variables"] for e in parsed_logs]
    # 2. build the variable matrix (alreay normalized matrix as it has values between 0 and 1)
    variables_distance_matrix = e.get_variable_matrix(parsed_variables).astype(float_precision)
    # 3. build the embeddings
    embeddings: List[np.ndarray] = [embedding for embedding in embedder(logs_texts)]
    # 4. build the distance matrix
    embeddings_distance_matrix = embedding_distance_fn(embeddings).astype(
        float_precision
    )
    del embeddings
    # 5. build the count matrix
    count_matrix = line_distance_fn(logs).astype(float_precision)
    # 6. merge the matrices with triplet coefficient
    combined_matrix = combine_matrices(
        TripletMatrix(
            variables_matrix=variables_distance_matrix,
            embeddings_matrix=embeddings_distance_matrix,
            count_matrix=count_matrix,
        ),
        triplet_coef=triplet_coefficient,
    ).astype(float_precision)
    # note: values will be between 0 and 3 (addition of 3 matrices normalized between 0 and 3)
    del variables_distance_matrix
    del embeddings_distance_matrix
    # 7. run the clustering algorithm with the constraints
    clustering_output = clustering_fn(combined_matrix)
    if isinstance(clustering_output, list):
        for i in range(len(clustering_output)):
            for coef,coef_val in triplet_coefficient.items():
                clustering_output[i]['hyperparameters'][coef] = coef_val
    else:
        for coef,coef_val in triplet_coefficient.items():
            clustering_output['hyperparameters'][coef] = coef_val
    # 8. return the result
    return clustering_output
