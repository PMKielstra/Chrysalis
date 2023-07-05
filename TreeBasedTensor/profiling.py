import numpy as np
from tensor import AK
from utils import subsample
from factor import UP, DOWN, BOTH

def ss_accuracy(A, compressed_AK, N):
    """Estimate the accuracy of compressed_AK as an estimation for A \times_{1, 2} K, subsampling the third and fourth axis to avoid running out of memory."""
    x = subsample(range(N))
    y = subsample(range(N))
    compressed_AK_subsampled = compressed_AK[x][:, y]
    true_AK_subsampled = AK(A, N, [list(range(N)), list(range(N)), x, y])
    return np.linalg.norm(compressed_AK_subsampled - true_AK_subsampled) / np.linalg.norm(true_AK_subsampled)

def matrix_memory(tree):
    """Determine the total amount of memory used by the matrices in a factor tree."""
    memory = sum(matrix_memory(child) for child in tree.children)
    if tree.rows_mats_up is not None:
        memory += sum(mat.size for _rows, mat in tree.rows_mats_up)
    if tree.rows_mats_down is not None:
        memory += sum(mat.size for _rows, mat in tree.rows_mats_down)
    return memory

def down_rows_length_dict(tree):
    """Get the number of tensor rows required at every leaf."""
    positions_dict = {}
    if tree.children == []:
        positions_dict[tree.position] = [len(rows) for rows, _mat in tree.rows_mats_down]
    else:
        for child in tree.children:
            positions_dict.update(down_rows_length_dict(child))
    return positions_dict

def leaf_sum(f, tree):
    """Sum the results of f over every leaf of tree."""
    if tree.children == []:
        return f(tree)
    return sum(leaf_sum(f, child) for child in tree.children)

def tensor_memory(factor_forest):
    """Determine the total amount of memory used by the tensors in a factor forest."""
    N, levels, off_cols_lists, trees, direction = factor_forest

    transpose_dicts = [down_rows_length_dict(tree) for tree in trees]
    off_cols_len = len(off_cols_lists[0]) // 2 ** levels

    if direction == DOWN:
        memory = 0
        for x in range(2 ** levels):
            for y in range(2 ** levels):
                memory += transpose_dicts[0][(x, y)][0] * N * off_cols_len * off_cols_len
        return memory
                
    elif direction == UP:
        return leaf_sum(lambda leaf: off_cols_len * off_cols_len * len(tree.rows_mats_up[0]) * N, trees[0])

    elif direction == BOTH:
        def size_transposed(col_split_position, up_leaf):
            assert len(up_leaf.rows_mats_up) == 2 ** levels
            
            x, y = up_leaf.position
            
            memory = 0
            for w in range(2 ** levels):
                up_rows, _up_mat = up_leaf.rows_mats_up[w]
                down_len = transpose_dicts[y][(w, col_split_position)][x]
                memory += down_len * off_cols_len * len(up_rows) * off_cols_len
            return memory
        return sum(leaf_sum(lambda leaf: size_transposed(i, leaf), tree) for i, tree in enumerate(trees))

    raise Exception(f"{direction} is not a valid direction!")

def total_memory(factor_forest):
    """Determine the total memory, both including and excluding tensor cores, used by a factor forest."""
    _N, _levels, _off_cols_lists, trees, _direction = factor_forest
    mm = sum(matrix_memory(tree) for tree in trees)
    return mm + tensor_memory(factor_forest), mm
