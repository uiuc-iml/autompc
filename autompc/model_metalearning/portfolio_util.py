import numpy as np
import pandas as pd

def normalize_matrix(matrix):
    normalized_matrix = matrix.copy()
    minima = np.nanmin(np.nanmin(normalized_matrix, axis=1), axis=0)
    maxima = np.nanmax(np.nanmax(normalized_matrix, axis=1), axis=0)
    print('min: {}; max: {}'.format(minima, maxima))
    diff = maxima - minima
    diff[diff == 0] = 1
    for task_idx in range(normalized_matrix.shape[0]):
        normalized_matrix[task_idx] = (
                (normalized_matrix[task_idx] - minima[task_idx]) / diff[task_idx]
        )

    assert (
        np.all((normalized_matrix >= 0) | (~np.isfinite(normalized_matrix)))
        and np.all((normalized_matrix <= 1) | (~np.isfinite(normalized_matrix)))
    ), (
        normalized_matrix, (normalized_matrix >= 0) | (~np.isfinite(normalized_matrix))
    )
    return normalized_matrix