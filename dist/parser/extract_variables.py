import numpy as np
import sklearn.preprocessing as skPrepro
import sklearn.metrics as skMetrics

from typing import *  # type: ignore

class EmptyLog(Exception):
    """Exception when no logs is found"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs


class NoVariable(Exception):
    """Exception when no variables are inside log lines"""

    def __init__(self, msg, logs: List[Dict[str, Any]]):
        super().__init__(msg)
        self.logs = logs

def get_variable_matrix(
    parsed_events: List[List[str]],
) -> np.ndarray:
    """Build the variable matrix from parsed logs

    # Arguments
    - parsed_events: List[List[str]], for each log line the variables inside this line. !! warning !! can be empty if there are no variable for a line

    # Returns
    - np.ndarray a one hot encoding indicating for each line if any of the variables inside the full log file is seen in this line
    """
    binarizer = skPrepro.MultiLabelBinarizer(sparse_output=False)
    matrix_variables = binarizer.fit_transform(parsed_events)
    if matrix_variables.shape[0] == 0:
        raise EmptyLog("No logs in the logs provided", logs=parsed_events)  # type: ignore
    if matrix_variables.shape[1] == 0:
        raise NoVariable("No variables in the logs provided", logs=parsed_events)  # type: ignore
    matrix_variables = matrix_variables.astype(bool)  # type: ignore
    matrix_variables_distance = skMetrics.pairwise_distances(
        matrix_variables, metric="jaccard"
    )
    return matrix_variables_distance
