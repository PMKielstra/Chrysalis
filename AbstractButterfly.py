from math import floor, ceil, log2
from copy import deepcopy
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

def tree_depth(bf, A, min_leaf_size, axes):
    return floor(log2(floor(min([bf.shape(A, axis) / min_leaf_size for axis in axes]))))

def list_transpose(l):
    return list(map(list, zip(*l)))

def single_axis_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, steps, depth):
    """Carry out a one-dimensional butterfly factorization along factor_axis, splitting along aux_axis."""

    def merge_list_doubles(l):
        return [bf.stack(l[2*i:2*i+2], axis=factor_axis) for i in range(len(l) // 2)]

    # Step 0: general setup
    factorization = bf.compose(None, None, None)
    singular_values_left = (factor_axis > aux_axis)

    # Step 1: factorization at leaf nodes
    leaves = bf.split(A, factor_axis, 2 ** depth)
    factored_leaves = list_transpose([bf.factor(leaf, factor_axis, aux_axis) for leaf in leaves])
    Us, Es = (factored_leaves[1], factored_leaves[0]) if singular_values_left else (factored_leaves[0], factored_leaves[1])
    factorization = bf.compose(factorization, bf.diag(Us), factor_axis) # Shortcut the U assembly
    
    # Step 2: setup for iteration
    E_blocks = [Es] # E = diag(map(merge, E_blocks))
    U_blocks = [Us]

    # Step 3: process a single E block
    def Es_to_Es_and_Rs(Es):
        split_Es = list_transpose([bf.split(E, aux_axis, 2) for E in merge_list_doubles(Es)])
        E_blocks = []
        R_cols = []
        for col in split_Es: # There should be two of these
            R_chunks = []
            E_col = []
            for E in col:
                factored_E = bf.factor(E, factor_axis, aux_axis)
                R, new_E = (factored_E[1], factored_E[0]) if singular_values_left else (factored_E[0], factored_E[1])
                R_chunks.append(R)
                E_col.append(new_E)
            R_cols.append(R_chunks)
            E_blocks.append(E_col)
        return E_blocks, R_cols

    def Us_and_Rs_to_Us(Us, Rs):
        diagonalized_Us = [bf.diag(Us[2*i:2*i+2]) for i in range(len(Us) // 2)]
        new_Us = []
        for R_col in Rs:
            new_Us.append([bf.multiply(R, U) if singular_values_left else bf.multiply(U, R) for U, R in zip(diagonalized_Us, R_col)])
        return new_Us

    # Step 4: process all the blocks
    for i in range(min(steps, depth)):
        new_U_blocks, new_E_blocks, Rs = [], [], []
        for E_block, U_block in zip(E_blocks, U_blocks):
            Es, R_cols = Es_to_Es_and_Rs(E_block)
            new_E_blocks += Es
            new_U_blocks += Us_and_Rs_to_Us(U_block, R_cols)
            R = bf.stack(list(map(bf.diag, R_cols)), axis=aux_axis)
            Rs.append(R)
        E_blocks = new_E_blocks
        U_blocks = new_U_blocks
        factorization = bf.compose(factorization, bf.diag(Rs), factor_axis)
    final_E_blocks = list(map(lambda E: bf.stack(E, axis=factor_axis), E_blocks))
    final_E = bf.diag(final_E_blocks)
    factorization_with_head = bf.compose(factorization, final_E, factor_axis)

    # Step 5: party!
    return factorization_with_head, factorization, U_blocks

def one_dimensional_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis):
    depth = tree_depth(bf, A, min_leaf_size, [factor_axis, aux_axis])
    full_factorization, _a, _b = single_axis_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, depth, depth)
    return full_factorization

def two_dimensional_butterfly(bf, A, min_leaf_size, axes):
    assert len(axes) == 2
    depth = tree_depth(bf, A, min_leaf_size, axes)
    _, left_factorization, left_U_blocks = single_axis_butterfly(bf, A, min_leaf_size, axes[0], axes[1], floor(depth / 2), depth)
    _, right_factorization, right_V_blocks = single_axis_butterfly(bf, A, min_leaf_size, axes[1], axes[0], ceil(depth / 2), depth)
    right_V_blocks = list_transpose(right_V_blocks)
    x, y = len(left_U_blocks), len(left_U_blocks[0])
    assert (x, y) == (len(right_V_blocks), len(right_V_blocks[0]))
    central_split = [bf.split(col, axes[0], y)\
                     for col in bf.split(A, axes[1], x)]
    central = []
    for i in range(x):
        row = []
        for j in range(y):
            U = (bf.transpose(left_U_blocks[i][j], axes[0], axes[1]))
            V = (bf.transpose(right_V_blocks[i][j], axes[0], axes[1]))
            UK = bf.build_center(central_split[i][j], U, \
                                 axes[0])
            UKV = bf.build_center(UK, V, axes[1])
            row.append(UKV)
        central.append(row)

    central_stacked = bf.diag(central, dimens=2)

    return bf.join(bf.compose(left_factorization, central_stacked, False), right_factorization)
