import numpy as np
import scipy as sp
import scipy.linalg.interpolative as interp
from tqdm import tqdm
from random import sample
from tensorly import unfold
from matplotlib import pyplot as plt
from functools import reduce

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

def ss_row_id(range1, range2, range3, range4, factor_rows):
    range3 = np_sample(range3)
    range4 = np_sample(range4)
    if factor_rows:
        range2 = np_sample(range2)
    else:
        range1 = np_sample(range1)
    ranged_K = lambda i, j, k, l: K_point(range1[i], range2[j], range3[k], range4[l])
    A = np.fromfunction(np.vectorize(ranged_K), (len(range1), len(range2), len(range3), len(range4)), dtype=int)
    unfolded = unfold(A, 0 if factor_rows else 1).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, eps)
    old_rows = np.array(range1 if factor_rows else range2)
    new_rows = old_rows[idx[:k]]
    R = interp.reconstruct_interp_matrix(idx, proj)
    return new_rows, R.T

def extend(range_x):
    full = np.array_split(list(range(N)), N // (len(range_x) * 2))
    return next(filter(lambda l: range_x[0] in l, full))

def compute_matrix(level, range1, range2, range3, range4, factor_rows):
    if level == 0:
        new_rows, R = ss_row_id(range1, range2, range3, range4, factor_rows)
        return new_rows, [R]
    if factor_rows:
        l = len(range1)//2
        rows_1, U_1 = compute_matrix(level - 1, range1[:l], range2, extend(range3), extend(range4), factor_rows)
        rows_2, U_2 = compute_matrix(level - 1, range1[l:], range2, extend(range3), extend(range4), factor_rows)
    else:
        l = len(range2)//2
        rows_1, U_1 = compute_matrix(level - 1, range1, range2[:l], extend(range3), extend(range4), factor_rows)
        rows_2, U_2 = compute_matrix(level - 1, range1, range2[l:], extend(range3), extend(range4), factor_rows)
    new_rows = np.concatenate((rows_1, rows_2))
    if factor_rows:
        new_rows, new_U = ss_row_id(new_rows, range2, range3, range4, factor_rows)
    else:
        new_rows, new_U = ss_row_id(range1, new_rows, range3, range4, factor_rows)
    return new_rows, [sp.linalg.block_diag(*UU) for UU in zip(U_1, U_2)] + [new_U]

rows, Us = compute_matrix(3, list(range(0, N)), list(range(0, N)), list(range(N // 8, N // 4)), list(range(N // 8, N // 4)), True)
full_U = reduce(np.matmul, Us)
A = np.fromfunction(np.vectorize(lambda i, j, k, l: K_point(rows[i], j, k + N // 8, l + N // 8)), (len(rows), N, N // 8, N // 8), dtype=int)
A = np.tensordot(full_U, A, axes=1)

def compare_with_real_K(A):
    B = np.fromfunction(np.vectorize(lambda i, j, k, l: K_point(i, j, k + N // 8, l + N // 8)), (N, N, N // 8, N // 8), dtype=int)
    return np.linalg.norm(A - B) / np.linalg.norm(A)

print(compare_with_real_K(A))
