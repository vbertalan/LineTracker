import numpy as np

from typing import *# type: ignore
from .main import LogData

def get_absolute_line_distance_matrix(
    events: List[LogData],
) -> np.ndarray:
    """Generate a matrix of shape (len(events),len(events)) where matrix[i,j] = abs(i-j)

    # Arguments
    - events: List[str], list of the text to generate the line distance matrix

    # Returns
    - np.ndarray, the distance matrix of shape (len(events),len(events)) where matrix[i,j] = abs(i-j)
    """
    n = len(events)
    matrix = np.array(
        [[abs(i - j) for j in range(n)] for i in range(n)], dtype=np.float32
    )
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))  # type: ignore
    return matrix