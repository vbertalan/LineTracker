"""Contains function to compute the distance matrix using line distance"""
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
    matrix = np.abs(np.arange(n)[:, np.newaxis] - np.arange(n))
    if np.max(matrix) == 0:
        return matrix
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))  # type: ignore
    return matrix