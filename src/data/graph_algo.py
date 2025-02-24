# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

def floyd_warshall(adjacency_matrix):
    # O(n^3) complexity
    nrows, ncols = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    # Copy adjacency matrix and convert to double precision
    M = adjacency_matrix.astype(np.float64, order='C', casting='safe', copy=True)
    path = np.zeros((n, n), dtype=np.int64)

    # Set unreachable nodes to infinity
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0
            elif M[i, j] == 0:
                M[i, j] = np.inf
    assert (np.diagonal(M) == 0.0).all()

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj
                    path[i, j] = k  # Save intermediate node

    # Set unreachable paths to 510
    M[M >= 510] = 510
    path[M >= 510] = 510

    return M, path

def get_all_edges(path, i, j):
    k = path[i, j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path, edge_feat):
    nrows, ncols = path.shape
    assert nrows == ncols
    n = nrows

    path_copy = path.astype(np.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(np.int64, order='C', casting='safe', copy=True)

    edge_fea_all = -1 * np.ones((n, n, max_dist, edge_feat.shape[-1]), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j or path_copy[i, j] == 510:
                continue

            # Reconstruct shortest path
            path_seq = [i] + get_all_edges(path_copy, i, j) + [j]
            path_len = len(path_seq) - 1
            for k in range(path_len):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path_seq[k], path_seq[k + 1], :]

    return edge_fea_all
