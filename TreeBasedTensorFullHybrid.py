import numpy as np
from scipy.linalg import block_diag
import scipy.linalg.interpolative as interp
from SliceManagement import Multirange, SliceTree
from tensorly import unfold
import itertools
import random
import matplotlib.pyplot as plt

N = 64
eps = 1e-6

def K_from_coords(coords_list):
    coords = np.meshgrid(*coords_list, indexing='ij')
    halflen = len(coords_list) // 2
    leftstack = np.stack(coords[:halflen], axis=0)
    rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
    return np.exp(1j * N * np.pi * norm) / norm

n_subsamples = 25
def np_sample(range_x):
    """Subsample a list at random."""
    return np.array(random.sample(list(range_x), min(n_subsamples, len(range_x))))

def ss_row_id(sampled_ranges, factor_index):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    subsamples = []
    for i, sr in enumerate(sampled_ranges):
        if i != factor_index and len(sr) > n_subsamples:
            subsamples.append(np_sample(sr))
        else:
            subsamples.append(sr)
    
    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = K_from_coords(subsamples)

    # Step 3: Unfold the tensor and carry out ID
    unfolded = unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, eps)
    R = interp.reconstruct_interp_matrix(idx, proj)

    # Step 4: Map the rows chosen by the ID, which are a subset of [1, ..., len(multirange[factor_index])], back to a subset of the relevant actual rows
    old_rows = sampled_ranges[factor_index]
    new_rows = old_rows[idx[:k]]
    return new_rows, R.T

class FactorTree:
    def __init__(self, rows_mats_down, rows_mats_up, position):
        self.rows_mats_down = rows_mats_down
        self.rows_mats_up = rows_mats_up
        self.position = position
        self.children = []

FACTOR_AXIS_SOURCE = 0
FACTOR_AXIS_OBSERVER = 2
POSITION_POWER = 4

def one_level_factor(rows_lists, off_cols, passive_multirange, is_source):
    if is_source:
        factor_ranges = [[r, off_cols] + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r, off_cols] for r in rows_lists]
    rows_mats = [ss_row_id(mr, FACTOR_AXIS_SOURCE if is_source else FACTOR_AXIS_OBSERVER) for mr in factor_ranges]
    new_rows = [np.concatenate((p[0], q[0])) for p, q in zip(rows_mats[::2], rows_mats[1::2])]
    return rows_mats, new_rows

def factor_to_tree(rows_lists_source, rows_lists_observer, off_cols, passive_multirange, level, position = 0):
    rows_mats_source, new_rows_source = one_level_factor(rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = one_level_factor(rows_lists_observer, off_cols, passive_multirange, is_source=False)
    tree = FactorTree(rows_mats_source, rows_mats_observer, position)
    if level > 0:
        position = position * POSITION_POWER
        child_results = [factor_to_tree(new_rows_source, new_rows_observer, off_cols, next_step, level - 1, position + i) for i, next_step in enumerate(passive_multirange.next_steps())]
        tree.children = [c[0] for c in child_results]
        leaf_list = []
        for c in child_results:
            leaf_list += c[1]
        return tree, leaf_list
    return tree, [tree]

def apply_down(split_A, tree, leaf_list):
    A_rows_mats_down = []
    for row_mat, A_1, A_2 in zip(tree.rows_mats_down, split_A[::2], split_A[1::2]):
        new_A = np.matmul(row_mat[1].T, np.concatenate((A_1, A_2)))
        A_rows_mats_down.append((row_mat[0], new_A))
    positions_dict = {}
    if tree.children == []:
        for i in range(len(A_rows_mats_down)):
            positions_dict[(tree.position, i)] = A_rows_mats_down[i]
    for child in tree.children:
        positions_dict.update(apply_down([Arm[1] for Arm in A_rows_mats_down], child, leaf_list))
    return positions_dict

def single_tree_bottom(positions_dict, down_cols, up_cols, tree, leaf_list):
    transposed_down_rows_mats = [positions_dict[(i, tree.position)] for i in range(len(leaf_list))]
    center_point_results = []
    for tree_row_mat, down_row_mat in zip(tree.rows_mats_up, transposed_down_rows_mats):
        K = K_from_coords((tree_row_mat[0], up_cols, down_row_mat[0], down_cols))
        center_point_results.append(tree_row_mat[1].dot(np.tensordot(K, down_row_mat[1], axes=2)))
    return np.concatenate(center_point_results)

def apply_up(positions_dicts_with_cols, off_cols, tree, leaf_list):
    if tree.children == []:
        return sum(single_tree_bottom(positions, down_cols, off_cols, tree, leaf_list) for positions, down_cols in positions_dicts_with_cols)
    split_As = [apply_up(positions_dicts_with_cols, off_cols, child, leaf_list) for child in tree.children]
    return block_diag(*([rm[1] for rm in tree.rows_mats_up])).dot(sum(split_As))

def build_factor_forest(levels, off_split_number):
    off_cols_lists = np.array_split(range(N), off_split_number)
    rows_list = np.array_split(range(N), 2 * POSITION_POWER ** levels) # TODO: Check formula.  Right now I only know it works with level=1.
    passive = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 2])
    trees_and_leaf_lists = [factor_to_tree(rows_list, rows_list, off_cols, passive, levels) for off_cols in off_cols_lists]
    return off_cols_lists, trees_and_leaf_lists

def apply(A, factor_forest):
    off_cols_lists, trees_and_leaf_lists = factor_forest
    transpose_dicts = [apply_down(np.array_split(A[:, cols], len(tll[1]) * 4), *tll) for cols, tll in zip(off_cols_lists, trees_and_leaf_lists)]
    split_KA = [apply_up(list(zip(transpose_dicts, off_cols_lists)), cols, *tll) for cols, tll in zip(off_cols_lists, trees_and_leaf_lists)]
    return np.concatenate(split_KA, axis=1)


A = np.random.rand(N, N)
factor_forest = build_factor_forest(1, 1)
compressed_A = apply(A, factor_forest)
true_A = np.tensordot(A, K_from_coords([list(range(N)), list(range(N)), list(range(N)), list(range(N))]), axes=((0, 1), (2, 3)))
print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(true_A))
