import numpy as np
from scipy.linalg import block_diag
import scipy.linalg.interpolative as interp
from SliceManagement import Multirange, SliceTree
from tensorly import unfold
import itertools
import matplotlib.pyplot as plt

N = 64
eps = 1e-4

def K_from_coords(coords_list):
    coords = np.meshgrid(*coords_list, indexing='ij')
    halflen = len(coords_list) // 2
    leftstack = np.stack(coords[:halflen], axis=0)
    rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
    return np.exp(1j * N * np.pi * norm) / norm

def ss_row_id(sampled_ranges, factor_index):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = K_from_coords(sampled_ranges)

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

def extend_to_sqrt(r):
    return r # TODO: COME UP WITH A SMART WAY OF DOING THIS

FACTOR_AXIS_SOURCE = 0
FACTOR_AXIS_OBSERVER = 2
POSITION_POWER = 4

def one_level_factor(rows_lists, passive_multirange, is_source):
    if is_source:
        factor_ranges = [[r, extend_to_sqrt(r)] + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r, extend_to_sqrt(r)] for r in rows_lists]
    rows_mats = [ss_row_id(mr, FACTOR_AXIS_SOURCE if is_source else FACTOR_AXIS_OBSERVER) for mr in factor_ranges]
    new_rows = [np.concatenate((p[0], q[0])) for p, q in zip(rows_mats[::2], rows_mats[1::2])]
    return rows_mats, new_rows

def factor_to_tree(rows_lists_source, rows_lists_observer, passive_multirange, level, position = 0):
    rows_mats_source, new_rows_source = one_level_factor(rows_lists_source, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = one_level_factor(rows_lists_observer, passive_multirange, is_source=False)
    tree = FactorTree(rows_mats_source, rows_mats_observer, position)
    if level > 0:
        position = position * POSITION_POWER
        child_results = [factor_to_tree(new_rows_source, new_rows_observer, next_step, level - 1, position + i) for i, next_step in enumerate(passive_multirange.next_steps())]
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

def apply_up(positions_dict, tree, leaf_list):
    if tree.children == []:
        transposed_down_rows_mats = [positions_dict[(i, tree.position)] for i in range(len(leaf_list))]
        center_point_results = []
        for tree_row_mat, down_row_mat in zip(tree.rows_mats_up, transposed_down_rows_mats):
            K = K_from_coords((tree_row_mat[0], list(range(N)), down_row_mat[0], list(range(N))))
            center_point_results.append(tree_row_mat[1].dot(np.tensordot(K, down_row_mat[1], axes=2)))
        return np.concatenate(center_point_results)
    split_As = [apply_up(positions_dict, child, leaf_list) for child in tree.children]
    return block_diag(*([rm[1] for rm in tree.rows_mats_up])).dot(sum(split_As))

def apply(A, tree, leaf_list):
    split_A = np.array_split(A, len(leaf_list) * 4)
    transpose_dict = apply_down(split_A, tree, leaf_list)
    return apply_up(transpose_dict, tree, leaf_list)

A = np.random.rand(N, N)
rows_list = np.array_split(list(range(N)), 8)
passive = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 2])
tree, leaf_list = factor_to_tree(rows_list, rows_list, passive, 1)
compressed_A = apply(A, tree, leaf_list)
true_A = true_A = np.tensordot(A, K_from_coords([list(range(N)), list(range(N)), list(range(N)), list(range(N))]), axes=((0, 1), (2, 3)))
print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(compressed_A))
