import numpy as np
from scipy.linalg import block_diag
import scipy.linalg.interpolative as interp
from SliceManagement import Multirange, SliceTree
from tensorly import unfold
import itertools
import matplotlib.pyplot as plt

N = 1024
eps = 1e-7

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
    def __init__(self, rows, matrix):
        self.matrix = matrix
        self.rows = rows
        self.children = []

def factor_to_tree(rows_list, multirange, factor_index, levels):
    new_rows, Rs = [], []
    for r in rows_list:
        row, R = ss_row_id(multirange.overwrite(SliceTree(r), factor_index), factor_index)
        new_rows.append(row)
        Rs.append(R)
    tree = FactorTree(np.concatenate(new_rows), block_diag(*Rs))
    merged_rows = [np.concatenate((a, b)) for a, b in zip(new_rows[::2], new_rows[1::2])]
    if levels > 0:
        tree.children = [factor_to_tree(merged_rows, step, factor_index, levels - 1) for step in multirange.next_steps()]
    return tree

def apply_down_tree(A, tree, factor_index):
    new_A = np.matmul(A, tree.matrix)
    if tree.children == []:
        return [(tree.rows, new_A)]
    return [i for l in (apply_down_tree(new_A, c, factor_index) for c in tree.children) for i in l]

def apply_up_tree(row_list, tree, factor_index):
    if tree.children == []:
        print(sum(len(r) * len(tree.rows) for r in row_list))
        matrix_row = [np.matmul(A, K_from_coords([r, tree.rows])) for r, A in row_list]
    else:
        matrix_row = [apply_up_tree(row_list, c, factor_index) for c in tree.children]
    matrix = sum(matrix_row) / len(matrix_row)
    return np.matmul(matrix, tree.matrix.T)

mr_rows = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 0])
mr_cols = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [0, 2])
split_range = np.array_split(list(range(N)), 128)

A = np.random.rand(1, N)

lst = apply_down_tree(A, factor_to_tree(split_range, mr_rows, 0, 3), 0)
compressed_K = apply_up_tree(lst, factor_to_tree(split_range, mr_cols, 1, 3), 1)
true_K = np.matmul(A, K_from_coords([list(range(N)), list(range(N))]))
plt.spy(compressed_K - true_K)
print(np.linalg.norm(compressed_K - true_K) / np.linalg.norm(true_K))
