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

def czip(*ls):
    for l in ls:
        assert len(l) == len(ls[0])
    return zip(*ls)

def K_from_coords(coords_list):
    coords = np.meshgrid(*coords_list, indexing='ij')
    halflen = len(coords_list) // 2
    leftstack = np.stack(coords[:halflen], axis=0)
    rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
    return np.exp(1j * N * np.pi * norm) / norm

n_subsamples = 32
def np_sample(range_x):
    """Subsample a list at random."""
    return np.array(random.sample(list(range_x), min(n_subsamples, len(range_x))))

def ss_row_id(sampled_ranges, factor_index):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    # Step 1: Subsample
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
        self.root = False

FACTOR_AXIS_SOURCE = 0
FACTOR_AXIS_OBSERVER = 2
POSITION_POWER = 4

def one_level_factor(rows_lists, off_cols, passive_multirange, is_source):
    if is_source:
        factor_ranges = [[r, off_cols] + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r, off_cols] for r in rows_lists]
    rows_mats = [ss_row_id(mr, FACTOR_AXIS_SOURCE if is_source else FACTOR_AXIS_OBSERVER) for mr in factor_ranges]
    if len(rows_mats) > 1:
        new_rows = [np.concatenate((p[0], q[0])) for p, q in czip(rows_mats[::2], rows_mats[1::2])]
    else:
        new_rows = rows_mats[0][0]
    return rows_mats, new_rows

def factor_to_tree(rows_lists_source, rows_lists_observer, off_cols, passive_multirange, level):
    rows_mats_source, new_rows_source = one_level_factor(rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = one_level_factor(rows_lists_observer, off_cols, passive_multirange, is_source=False)
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange.position())
    if level > 0:
        child_results = [factor_to_tree(new_rows_source, new_rows_observer, off_cols, next_step, level - 1) for i, next_step in enumerate(passive_multirange.next_steps())]
        tree.children = [c[0] for c in child_results]
        leaf_list = []
        for c in child_results:
            leaf_list += c[1]
        return tree, leaf_list
    return tree, [tree]

def apply_down(split_A, tree, leaf_list):
    A_rows_mats_down = []
    if tree.root:
        for row_mat, A in czip(tree.rows_mats_down, split_A):
            new_A = np.matmul(row_mat[1].T, A)
            A_rows_mats_down.append((row_mat[0], new_A))
    else:
        for row_mat, A_1, A_2 in czip(tree.rows_mats_down, split_A[::2], split_A[1::2]):
            new_A = np.matmul(row_mat[1].T, np.concatenate((A_1, A_2)))
            A_rows_mats_down.append((row_mat[0], new_A))
    positions_dict = {}
    if tree.children == []:
        positions_dict[tree.position] = []
        for i in range(len(A_rows_mats_down)):
            positions_dict[tree.position].append((A_rows_mats_down[i][0], A_rows_mats_down[i][1]))
    for child in tree.children:
        positions_dict.update(apply_down([Arm[1] for Arm in A_rows_mats_down], child, leaf_list))
    return positions_dict

def apply_up(tree, get_transposed_split_As):
    if tree.children == []:
        return np.concatenate(get_transposed_split_As(tree), axis=0)
    split_As = [apply_up(child, get_transposed_split_As) for child in tree.children]
    return block_diag(*([rm[1] for rm in tree.rows_mats_up])).dot(sum(split_As))

def build_factor_forest(levels, one_way=False):
    off_split_number = 1 if one_way else 2 ** levels
    off_cols_lists = np.array_split(range(N), off_split_number)
    rows_list = np.array_split(range(N), 2 ** (levels if one_way else 2 * levels))
    passive = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 2])
    trees_and_leaf_lists = [factor_to_tree(rows_list, rows_list, off_cols, passive, levels) for off_cols in off_cols_lists]
    for t in trees_and_leaf_lists:
        t[0].root = True
    return levels, off_cols_lists, trees_and_leaf_lists, one_way

def apply(A, factor_forest):
    levels, off_cols_lists, trees_and_leaf_lists, one_way = factor_forest
    transpose_dicts = [apply_down(np.split(A[:, cols], 2 ** (levels if one_way else 2 * levels)), *tll) for cols, tll in czip(off_cols_lists, trees_and_leaf_lists)]

    if one_way:
        cols = off_cols_lists[0]
        cols_split = np.array_split(cols, 2 ** levels)
        positions_dict = transpose_dicts[0]
        block = []
        for x in range(len(cols_split)):
            row = []
            for y in range(len(cols_split)):
                down_rows, down_mat = positions_dict[(x, y)][0]
                down_cols = list(range(N))
                up_rows = cols_split[x]
                up_cols = cols_split[y]
                K = K_from_coords((down_rows, down_cols, up_rows, up_cols))
                AK = np.tensordot(down_mat, K, axes=2)
                row.append(AK)
            block.append(row)
        return np.block(block)

    def get_transposed_KA_leaf(col_split_position, up_leaf):
        split_As = []
        x, y = up_leaf.position
        up_cols = off_cols_lists[col_split_position]
        down_cols = off_cols_lists[y]
        split_As = []
        assert len(up_leaf.rows_mats_up) == 2 ** levels
        for w in range(2 ** levels):
            up_rows, up_mat = up_leaf.rows_mats_up[w]
            down_rows, down_mat = transpose_dicts[y][(w, col_split_position)][x]
            K = K_from_coords((down_rows, down_cols, up_rows, up_cols))
            AK = np.tensordot(down_mat, K, axes=2)
            split_As.append(up_mat.dot(AK))
        return split_As

    split_KA = [apply_up(tll[0], lambda leaf: get_transposed_KA_leaf(i, leaf)) for i, tll in enumerate(trees_and_leaf_lists)]
    return np.concatenate(split_KA, axis=1)

def just_apply_up(A, factor_forest):
    levels, off_cols_lists, trees_and_leaf_lists, one_way = factor_forest
    assert one_way

    cols = off_cols_lists[0]
    cols_split = np.array_split(cols, 2 ** levels)
        

    def get_transposed_KA_leaf(up_leaf):
        x, y = up_leaf.position
        down_rows, down_cols = cols_split[x], cols_split[y]
        up_cols = list(range(N))
        assert len(up_leaf.rows_mats_up) == 1
        up_rows, up_mat = up_leaf.rows_mats_up[0]
        K = K_from_coords((down_rows, down_cols, up_rows, up_cols))
        AK = np.tensordot(A[down_rows][:, down_cols], K, axes=2)
        return [up_mat.dot(AK)]
    
    split_KA = apply_up(trees_and_leaf_lists[0][0], get_transposed_KA_leaf)
    return split_KA

A = np.random.rand(N, N)
factor_forest = build_factor_forest(3)
compressed_A = apply(A, factor_forest)
true_A = np.tensordot(A, K_from_coords([list(range(N)), list(range(N)), list(range(N)), list(range(N))]), axes=2)
print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(true_A))
