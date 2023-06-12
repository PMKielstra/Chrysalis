import numpy as np
import scipy as sp
import scipy.linalg.interpolative as interp
from tqdm import tqdm
from random import sample
from tensorly import unfold
from matplotlib import pyplot as plt
from functools import reduce

from SliceManagement import SliceTree, split_to_tree, Multirange, EXTEND, IGNORE

N = 32
eps = 1e-6

xs = np.linspace(0, 1, N)
xgrid, ygrid = np.meshgrid(xs, xs)

def K(r1, r2):
    """Tensor kernel representing the interaction between the two points r1 and r2."""
    d = np.linalg.norm(r1 - r2)
    return np.exp(1j * N * np.pi * d) / d

def K_point(i, j, k, l):
    """Tensor kernel at a point -- (i, j) are considered as points on the source grid, with z=0, and (k, l) as points on the observer grid, with z=1.  Mapping from [0, N] to [0, 1] is performed automatically.  Expects integer arguments."""
    return K(np.array((0, xgrid[i, j], ygrid[i, j])), np.array((1, xgrid[k, l], ygrid[k, l])))

def make_K_from_list(points):
    """Create a version of K_point which can be used with arbitrary subsets of the usual grid rows and columns."""
    return lambda *pos: K_point(*(p[q] for p, q in zip(points, pos)))

n_subsamples = 20
def np_sample(range_x):
    """Subsample a list at random."""
    return np.array(sample(list(range_x), min(n_subsamples, len(range_x))))

def ss_row_id(multirange, factor_index):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    
    # Step 1: Subsample all rows but the factor index
    sample_apply = [np_sample] * len(multirange)
    sample_apply[factor_index] = lambda x: x
    sampled_ranges = multirange.apply(sample_apply)

    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = np.fromfunction(np.vectorize(make_K_from_list(sampled_ranges)), (len(r) for r in sampled_ranges), dtype=int)

    # Step 3: Unfold the tensor and carry out ID
    unfolded = unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, eps)
    R = interp.reconstruct_interp_matrix(idx, proj)

    # Step 4: Map the rows chosen by the ID, which are a subset of [1, ..., len(multirange[factor_index])], back to a subset of the relevant actual rows
    old_rows = np.array(multirange[factor_index])
    new_rows = old_rows[idx[:k]]
    return new_rows, R.T

def compute_matrices(level, multirange, factor_index):
    """Carry out a multi-level, single-axis butterfly, returning a subset of rows along that axis along with a list of interpolation matrices."""
    if level == 0:
        new_rows, R = ss_row_id(multirange, factor_index)
        return new_rows, [R]
    next_steps = multirange.next_steps()
    next_results = [compute_matrices(level - 1, step, factor_index) for step in next_steps]
    new_rows = np.concatenate([row for row, U in next_results])
    new_rows, new_U = ss_row_id(multirange.overwrite(new_rows, factor_index), factor_index)
    return new_rows, [sp.linalg.block_diag(*UU) for UU in zip(*(U for row, U in next_results))] + [new_U]

def factor_chunk(x_index, y_index, levels, splits_per_level, factor_index):
    assert factor_index in [0, 1] # There's a much more general solution here, but no slightly-more-general solution, so it's not worth the hassle.
    x_split_tree = split_to_tree(list(range(N)), levels, splits_per_level)[x_index]
    y_split_tree = split_to_tree(list(range(N)), levels, splits_per_level)[y_index]
    split_pattern = [IGNORE, IGNORE, EXTEND, EXTEND]
    split_pattern[factor_index] = splits_per_level
    section_ranges = Multirange([
            SliceTree(list(range(N))),
            SliceTree(list(range(N))),
            x_split_tree,
            y_split_tree
        ],
        split_pattern)
    return compute_matrices(levels, section_ranges, factor_index)

def evaluate_chunk(x_index, y_index, levels, splits_per_level, factor_index):
    assert factor_index in [0, 1]
    rows, Us = factor_chunk(x_index, y_index, levels, splits_per_level, factor_index)
    full_U = reduce(np.matmul, Us)
    x_split_tree = split_to_tree(list(range(N)), levels, splits_per_level)[x_index]
    y_split_tree = split_to_tree(list(range(N)), levels, splits_per_level)[y_index]
    points = [list(range(N)), list(range(N)), x_split_tree[:], y_split_tree[:]]
    true_A = np.fromfunction(np.vectorize(make_K_from_list(points)), (len(p) for p in points), dtype=int)
    points[factor_index] = rows
    compressed_A = np.fromfunction(np.vectorize(make_K_from_list(points)), (len(p) for p in points), dtype=int)
    decompressed_A = np.swapaxes(np.tensordot(full_U, compressed_A, axes=((1,), (factor_index,))), 0, factor_index)
    return np.linalg.norm(true_A - decompressed_A) / np.linalg.norm(true_A)

print(evaluate_chunk(1, 2, 2, 2, 1))
