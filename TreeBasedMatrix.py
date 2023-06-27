import numpy as np
from scipy.linalg import block_diag
import scipy.linalg.interpolative as interp
from SliceManagement import Multirange, SliceTree
from tensorly import unfold
import itertools
import matplotlib.pyplot as plt

N = 1024
eps = 1e-10

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
    def __init__(self, rows_mats):
        self.rows_mats = rows_mats
        self.children = []
        self.down_results = []

def factor_to_tree(rows_lists, multirange, factor_axis, level):
    rows_mats = [ss_row_id(multirange.overwrite(SliceTree(r), factor_axis), factor_axis) for r in rows_lists]
    new_rows = [np.concatenate((p[0], q[0])) for p, q in zip(rows_mats[::2], rows_mats[1::2])]
    tree = FactorTree(rows_mats)
    if level > 0:
        tree.children = [factor_to_tree(new_rows, next_range, factor_axis, level - 1) for next_range in multirange.next_steps()]
    return tree

def apply_down(split_A, tree, factor_axis):
    results = []
    for row_mat, A_1, A_2 in zip(tree.rows_mats, split_A[::2], split_A[1::2]):
        new_A = np.matmul(row_mat[1].T, np.concatenate((A_1, A_2)))
        results.append((row_mat[0], new_A))
    if tree.children == []:
        return [results]
    child_results = [apply_down([r[1] for r in results], child, factor_axis) for child in tree.children]
    return [c for sublist in child_results for c in sublist]

def list_transpose(l):
    return [[ll[i] for ll in l] for i in range(len(l[0]))]

def propagate_down(transposed_lists, tree):
    if tree.children == []:
        assert len(transposed_lists) == 1
        tree.down_results = [K_from_coords((tree_rows_mats[0], down_rows_mats[0])).dot(down_rows_mats[1]) for tree_rows_mats, down_rows_mats in zip(tree.rows_mats, transposed_lists[0])]
        return
    for i, child in enumerate(tree.children):
        propagate_down(transposed_lists[i*(len(transposed_lists) // len(tree.children)) : (i+1) * (len(transposed_lists) // len(tree.children))], child)

def propagate_up(tree):
    if tree.children == []:
        split_As = tree.down_results
        return np.concatenate([rm[1].dot(A) for rm, A in zip(tree.rows_mats, split_As)])
    split_As = [propagate_up(child) for child in tree.children]
    return block_diag(*[rm[1] for rm in tree.rows_mats]).dot(sum(split_As))


mr_rows = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [0, 2])
mr_cols = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 0])
split_ranges = np.array_split(list(range(N)), 256)
A = np.random.rand(N)
split_A = np.array_split(A, 512)

tree_rows = factor_to_tree(split_ranges, mr_rows, 0, 4)
k = list_transpose(apply_down(split_A, tree_rows, 0))

tree_cols = factor_to_tree(split_ranges, mr_cols, 1, 4)
propagate_down(k, tree_cols)
compressed_A = propagate_up(tree_cols)

true_A = K_from_coords([list(range(N)), list(range(N))]).dot(A)

print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(true_A))

