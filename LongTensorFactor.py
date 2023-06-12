import numpy as np
import scipy as sp
import scipy.linalg.interpolative as interp
from tqdm import tqdm
from random import sample
from tensorly import unfold
from matplotlib import pyplot as plt
from functools import reduce

from slicetree import SliceTree, split_to_tree, Multirange, EXTEND, IGNORE

N = 32
eps = 1e-6

xs = np.linspace(0, 1, N)
xgrid, ygrid = np.meshgrid(xs, xs)

def K(r1, r2):
    return np.exp(1j * N * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def K_point(i, j, k, l):
    return K(np.array((0, xgrid[i, j], ygrid[i, j])), np.array((1, xgrid[k, l], ygrid[k, l])))

n_subsamples = 20
def np_sample(range_x):
    return np.array(sample(list(range_x), min(n_subsamples, len(range_x))))

def ss_row_id(multirange, factor_index):
    sample_apply = [np_sample] * len(multirange)
    sample_apply[factor_index] = lambda x: x
    sampled_ranges = multirange.apply(sample_apply)
    ranged_K = lambda *pos: K_point(*(r[q] for r, q in zip(sampled_ranges, pos)))
    A = np.fromfunction(np.vectorize(ranged_K), (len(r) for r in sampled_ranges), dtype=int)
    unfolded = unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, eps)
    old_rows = np.array(multirange[factor_index])
    new_rows = old_rows[idx[:k]]
    R = interp.reconstruct_interp_matrix(idx, proj)
    return new_rows, R.T

def compute_matrix(level, multirange, factor_index):
    if level == 0:
        new_rows, R = ss_row_id(multirange, factor_index)
        return new_rows, [R]
    next_steps = multirange.next_steps()
    rows_1, U_1 = compute_matrix(level - 1, next_steps[0], factor_index)
    rows_2, U_2 = compute_matrix(level - 1, next_steps[1], factor_index)
    new_rows = np.concatenate((rows_1, rows_2))
    new_rows, new_U = ss_row_id(multirange.overwrite(new_rows, factor_index), factor_index)
    return new_rows, [sp.linalg.block_diag(*UU) for UU in zip(U_1, U_2)] + [new_U]

section_ranges = Multirange([SliceTree(list(range(0, N))), SliceTree(list(range(0, N))), split_to_tree(list(range(N)), 3, 2)[1], split_to_tree(list(range(N)), 3, 2)[1]], (2, IGNORE, EXTEND, EXTEND))

rows, Us = compute_matrix(3, section_ranges, 0)
full_U = reduce(np.matmul, Us)
A = np.fromfunction(np.vectorize(lambda i, j, k, l: K_point(rows[i], j, k + N // 8, l + N // 8)), (len(rows), N, N // 8, N // 8), dtype=int)
A = np.tensordot(full_U, A, axes=1)

def compare_with_real_K(A):
    B = np.fromfunction(np.vectorize(lambda i, j, k, l: K_point(i, j, k + N // 8, l + N // 8)), (N, N, N // 8, N // 8), dtype=int)
    return np.linalg.norm(A - B) / np.linalg.norm(A)

print(compare_with_real_K(A))
