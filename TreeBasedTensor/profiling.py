import itertools

import numpy as np

from tensor import AK
from utils import subsample, slice_by_index, multilevel_access, multilevel_flatten, multilevel_enumerate
from factor import UP, DOWN, BOTH
from multiwaymatvec import make_grid, forest_children

accuracy_subsamples = 10
def ss_accuracy(profile, A, compressed_AK):
    """Estimate the accuracy of compressed_AK as an estimation for A \times_{1, 2} K, subsampling the third and fourth axis to avoid running out of memory."""
    if profile.as_matrix:
        A = np.ravel(A)
    subsample_ranges = [subsample(range(profile.N), accuracy_subsamples) for _ in range(profile.dimens)]
    compressed_AK_subsampled = slice_by_index(compressed_AK, subsample_ranges)
    true_AK_subsampled = AK(profile, A, [list(range(profile.N))] * profile.dimens + subsample_ranges)
    return np.linalg.norm(compressed_AK_subsampled - true_AK_subsampled) / np.linalg.norm(true_AK_subsampled)

def factor_forests_fold(f, factor_forests, combine):
    """Calls f on every tree in factor_forests and then combines the results with combine."""
    if isinstance(factor_forests, list):
        return combine(factor_forests_fold(f, forest, combine) for forest in factor_forests)
    return f(factor_forests)

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

def make_base_dict(profile, factor_forests, level):
    test_tree = multilevel_access(factor_forests[0], [0] * (profile.dimens - 1))

    if test_tree.children == []:
        return {test_tree.position: [make_grid(profile, forest, axis, 2 ** level, rows_grid=True) for axis, forest in enumerate(factor_forests)]}

    base_dict = {}
    for c in forest_children(profile, factor_forests):
        base_dict.update(make_base_dict(profile, c, level - 1))
    return base_dict

def tensor_memory(profile, factor_forests):
    """Determine the total amount of memory used by the tensors in a factor forest."""

    base_dict = make_base_dict(profile, factor_forests, profile.levels)

    def size_at(down_position, up_position):
        down_up_rows_lists = base_dict[down_position]
        down_rows_lists = [du(up_position)[0] for du in down_up_rows_lists]
        if profile.direction == BOTH:
            down_up_rows_lists = base_dict[up_position]
            up_rows_lists = [du(down_position)[1] for du in down_up_rows_lists]
        else:
            up_rows_lists = [range(profile.N // 2 ** profile.levels)] * profile.dimens
        lengths = [len(l) for l in down_rows_lists + up_rows_lists]
        return np.prod(lengths)

    down_positions = [tuple(l) for l in itertools.product(*[range(2 ** profile.levels)] * profile.dimens)]

    if profile.direction == BOTH:
        up_positions = down_positions
    else:
        up_positions = [tuple([0] * profile.dimens)]
    return sum(size_at(down, up) for down, up in itertools.product(down_positions, up_positions))

def total_memory(profile, factor_forests):
    """Determine the total memory, both including and excluding tensor cores, used by a set of factor forests."""
    mm = factor_forests_fold(matrix_memory, factor_forests, sum)
    tm = tensor_memory(profile, factor_forests)
    return mm + tm, mm

def max_leaf_row_length(tree):
    """Determine the maximum length of the list of rows (aka the maximum rank) at any leaf of a tree."""
    if tree.children == []:
        return max(0 if tree.rows_mats_up is None else max(len(rows) for rows, _mat in tree.rows_mats_up),
                   0 if tree.rows_mats_down is None else max(len(rows) for rows, _mat in tree.rows_mats_down))
    return max(max_leaf_row_length(child) for child in tree.children)

def max_leaf_row_length_forests(factor_forests):
    """Determine the maximum length of the list of rows (aka the maximum rank) at any leaf of any tree in a set of forests."""
    return factor_forests_fold(max_leaf_row_length, factor_forests, max)
                   
