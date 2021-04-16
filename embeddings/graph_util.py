# Utility methods for managing graphs

# Author : Shikhar Tuli

import numpy as np
import hashlib


def model_dict_to_graph(model_dict, ops_list):
    """Converts model_dict to model_graph which is a tuple of
    adjacency matrix and ops

    Args:
        model_dict: the model dictionary
        ops_list: list of all possible operations

    Returns:
        model_graph = (adjacency_matrix, ops)

    Raises:
        ValueError: if a required operation is not in ops_list
    """
    layers = model_dict.get('l')
    hidden = model_dict.get('h')
    attention = model_dict.get('a')
    feed_forward = model_dict.get('f')
    similarity = model_dict.get('s')

    assert layers == len(hidden) == len(attention) == len(feed_forward) == len(similarity), \
        f'Input model_dict is incorrect:\n{model_dict}'

    V = 1 # One vertex for the input
    for layer in range(layers):
        # Input goes into a few attention heads
        # Then, the output of multi-head attention goes to add-norm
        # This then goes to a feed-forward layer
        # Finally, we have the last add-norm layer after the feed-forward layer
        V += attention[layer] + 1 + 1 + 1
    V += 1 # Finally, the output vertex

    matrix = np.zeros((V, V), dtype=int)

    # Generate operations list
    ops = ['input']
    for layer in range(layers): 
        for head in range(attention[layer]):
            op = f'a_h{hidden[layer]}_s-{similarity[layer]}'
            if op not in ops_list:
                raise ValueError(f'Operation: {op}, not in ops_list')
            else:
                ops.append(op)
        op = f'f{feed_forward[layer]}'
        if op not in ops_list:
            raise ValueError(f'Operation: {op}, not in ops_list')
        ops.extend(['add_norm', op, 'add_norm'])
    ops.append('output')

    assert len(ops) == V, \
        f'Number of operations is not equal to size of the adjacency matrix:\nOperations: {ops}\nV: {V}'

    # Generate adjacency matrix
    i = 0
    while i < len(ops) - 2:
        for layer in range(layers):
            for h in range(attention[layer]):
                matrix[i][i + 1 + h] = 1 # Input to all attention heads
                matrix[i + 1 + h][i + 1 + attention[layer]] = 1 # attention head to add_norm
            i += attention[layer] + 1 # Re-instate i to add_norm
            matrix[i][i + 1] = 1 # add_norm to feed_forward
            matrix[i + 1][i + 2] = 1 # feed_forward to add_norm
            i += 2 # Re-instate i to add_norm as output of current encoder layer
    matrix[-2][-1] = 1 # add_norm to output
 
    assert check_upper_tri(matrix), \
        f'Matrix generated is not an upper-triangular adjacency matrix: \n{matrix}'
    assert is_full_dag(matrix), \
        f'Matrix generated has "hanging" vertices: \n{matrix}'

    return (matrix, ops)


def check_upper_tri(matrix):
    """Checks if the given matrix is an upper-triangular adjacency matrix
    
    Args:
        matrix: A numpy matrix

    Returns:
        True if matrix is an upper-triangular adjacency matrix
    """
    return np.allclose(matrix, np.triu(matrix)) \
        and np.amax(matrix) == 1 \
        and np.amin(matrix) == 0


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).
    i.e. no disconnected or "hanging" vertices.

    It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

    Args:
        matrix: V x V upper-triangular adjacency matrix

    Returns:
        True if the there are no dangling vertices.
    """
    shape = np.shape(matrix)

    rows = matrix[:shape[0]-1, :] == 0
    rows = np.all(rows, axis=1)     # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)     # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_func(str, algo='md5'):
  """Outputs has based on algorithm defined."""
  return eval(f"hashlib.{algo}(str)")


def hash_graph(matrix, ops, algo='md5'):
    """Computes a graph-invariant hash of the matrix and ops pair.

    Args:
        matrix: np.ndarray square upper-triangular adjacency matrix.
        ops: list operations of length equal to both dimensions of
            matrix.
        algo: hash algorithm among ["md5", "sha256", "sha512"]

    Returns:
        Hash of the matrix and ops based on algo.
    """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(ops)
    hashes = list(zip(out_edges, in_edges, ops))
    hashes = [hash_func(str(h).encode('utf-8'), algo).hexdigest() for h in hashes]

    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hash_func(
                (''.join(sorted(in_neighbors)) + '|' +
                ''.join(sorted(out_neighbors)) + '|' +
                hashes[v]).encode('utf-8'), algo).hexdigest())
        hashes = new_hashes

    fingerprint = hash_func(str(sorted(hashes)).encode('utf-8'), algo).hexdigest()

    return fingerprint