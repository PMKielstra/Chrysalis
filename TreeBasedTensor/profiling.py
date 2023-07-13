import numpy as np
from tensor import AK
from utils import subsample, slice_by_index, multilevel_access, multilevel_flatten, multilevel_enumerate
from factor import UP, DOWN, BOTH

accuracy_subsamples = 10
def ss_accuracy(profile, A, compressed_AK):
    """Estimate the accuracy of compressed_AK as an estimation for A \times_{1, 2} K, subsampling the third and fourth axis to avoid running out of memory."""
    if profile.as_matrix:
        A = np.ravel(A)
    subsample_ranges = [subsample(range(profile.N), accuracy_subsamples) for _ in range(profile.dimens)]
    compressed_AK_subsampled = slice_by_index(compressed_AK, subsample_ranges)
    true_AK_subsampled = AK(profile, A, [list(range(profile.N))] * profile.dimens + subsample_ranges)
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

def tensor_memory(profile, factor_forest):
    """Determine the total amount of memory used by the tensors in a factor forest."""
    off_cols_lists, trees = factor_forest

    def build_transpose_dict(trees, i):
        if i == profile.dimens:
            return down_rows_length_dict(trees)
        return [build_transpose_dict(tree, i + 1) for tree in trees]

    transpose_dicts = build_transpose_dict(trees, 1)
    off_cols_len = len(off_cols_lists[0]) // 2 ** profile.levels

    if profile.direction == DOWN:
        def block(dimen, position):
            if dimen == 0:
                return multilevel_access(transpose_dicts, [0] * (profile.dimens - 1), assert_single_element=True)[tuple(position)][0] * (profile.N ** (profile.dimens - 1)) * (off_cols_len ** profile.dimens)
            return sum(block(dimen - 1, position + [i]) for i in range(2 ** profile.levels))
        return block(profile.dimens, [])

    elif profile.direction == UP:
        return leaf_sum(lambda leaf: (off_cols_len ** profile.dimens) * len(leaf.rows_mats_up[0][0]) * (profile.N ** (profile.dimens - 1)), multilevel_access(trees, [0] * (profile.dimens - 1)))

    elif profile.direction == BOTH:
        def size_transposed(col_split_position, up_leaf):
            assert len(up_leaf.rows_mats_up) == 2 ** profile.levels
                        
            memory = 0
            for w in range(2 ** profile.levels):
                up_rows, _up_mat = up_leaf.rows_mats_up[w]
                down_len = multilevel_access(transpose_dicts, up_leaf.position[1:])[tuple([w] + col_split_position)][up_leaf.position[0]]
                memory += down_len * (off_cols_len ** (profile.dimens - 1)) * len(up_rows) * (off_cols_len ** (profile.dimens - 1))
            return memory
        return sum(leaf_sum(lambda leaf: size_transposed(i, leaf), tree) for i, tree in multilevel_enumerate(trees, profile.dimens - 1))

    raise Exception(f"{direction} is not a valid direction!")

def total_memory(profile, factor_forest):
    """Determine the total memory, both including and excluding tensor cores, used by a factor forest."""
    _off_cols_lists, trees = factor_forest
    mm = sum(matrix_memory(tree) for tree in multilevel_flatten(trees))
    return mm + tensor_memory(profile, factor_forest), mm

def max_leaf_row_length(tree):
    """Determine the maximum length of the list of rows (aka the maximum rank) at any leaf of a tree."""
    if tree.children == []:
        return max(0 if tree.rows_mats_up is None else max(len(rows) for rows, _mat in tree.rows_mats_up),
                   0 if tree.rows_mats_down is None else max(len(rows) for rows, _mat in tree.rows_mats_down))
    return max(max_leaf_row_length(child) for child in tree.children)

def max_leaf_row_length_forest(factor_forest):
    """Determine the maximum length of the list of rows (aka the maximum rank) at any leaf of any tree in a forest."""
    _off_cols_lists, trees = factor_forest
    return max(max_leaf_row_length(tree) for tree in multilevel_flatten(trees))
                   
