from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids = []
    col_ids = []
    M, N = cost_matrix.shape[0], cost_matrix.shape[1]
    cost_mat = np.copy(cost_matrix)
    S1 = np.linspace(0, M-1, M).astype(int)
    S2 = np.linspace(0, N-1, N).astype(int)
    for _ in range(min(M, N)):
        # print("======================")
        # print(S1)
        # print(S2)
        # print("======================")
        ids = np.array(np.meshgrid(S1, S2)).T.reshape(-1, 2)    # S1xS2
        all_ids = np.where(cost_mat == np.amin(cost_mat[ids[:, 0], ids[:, 1]]))   # find all smallest element ind
        all_ids_lst = list(zip(all_ids[0], all_ids[1]))
        assignment = -1
        for i in all_ids_lst:
            if i[0] in S1 and i[1] in S2:
                assignment = i
                break
        if assignment == -1:
            pass
        row_ids.append(assignment[0])
        col_ids.append(assignment[1])
        S1 = np.delete(S1, np.argwhere(S1 == assignment[0]))
        S2 = np.delete(S2, np.argwhere(S2 == assignment[1]))
    return row_ids, col_ids


def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return row_ids, col_ids
