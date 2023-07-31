from copy import deepcopy

import numpy as np
import scipy as sp

from utils import without, czip, slice_by_index, multilevel_access
from profile import BOTH, UP, DOWN
from tensor import AK
from tensorgrid import TensorGrid

def make_grid(profile, factor_forest, axis, off_multiplication, rows_grid=False, up=False, merge_level=0):
    def get_U_at(position):
        position = list(position)
        off_position = without(position, axis)
        off_position = [p // off_multiplication for p in off_position]
        factor_tree = multilevel_access(factor_forest, off_position)
        if up:
            merge = 2 ** merge_level
            return sp.linalg.block_diag(*[r[1] for r in factor_tree.rows_mats_up[merge * position[axis] : merge * position[axis] + merge]])
        if rows_grid:
            return factor_tree.rows_mats_down[position[axis]][0], factor_tree.rows_mats_up[position[axis]][0]
        return factor_tree.rows_mats_down[position[axis]][1].T
    return get_U_at

def binary_swap_axes(c, axis, dimens):
    binary_c = [int(digit) for digit in bin(c)[2:]]
    while len(binary_c) < dimens:
        binary_c = [0] + binary_c
    binary_c[0], binary_c[axis] = binary_c[axis], binary_c[0]
    out_c = 0
    for bit in binary_c:
        out_c = (out_c << 1) | bit
    return out_c

def forest_child_at(forest, child_position, axis, dimens):
    if isinstance(forest, list):
        return [forest_child_at(f, child_position, axis, dimens) for f in forest]
    return forest.children[binary_swap_axes(child_position, axis, dimens)]

def forest_children(profile, factor_forests):
    return [[forest_child_at(factor_forest, c, axis, profile.dimens) for axis, factor_forest in enumerate(factor_forests)] for c in range(2 ** profile.dimens)]

def apply_down(profile, A_grid, factor_forests, level):
    test_tree = multilevel_access(factor_forests[0], [0] * (profile.dimens - 1))
    if not test_tree.root:
        A_grid.merge_once()
    for axis, forest in enumerate(factor_forests):
        U_grid = make_grid(profile, forest, axis, 2 ** level)
        A_grid.axis_multiply(U_grid, axis)

    if test_tree.children == []:
        return {test_tree.position: ([make_grid(profile, forest, axis, 2 ** level, rows_grid=True) for axis, forest in enumerate(factor_forests)], deepcopy(A_grid))}

    base_dict = {}
    for c in forest_children(profile, factor_forests):
        base_dict.update(apply_down(profile, deepcopy(A_grid), c, level - 1))
    return base_dict

def apply_up(profile, get_leaf_grid, factor_forests, level):
    test_tree = multilevel_access(factor_forests[0], [0] * (profile.dimens - 1))
    if test_tree.children == []:
        leaf = get_leaf_grid(test_tree.position)
    else:
        children = forest_children(profile, factor_forests)
        downward_leaves = [apply_up(profile, get_leaf_grid, child, level + 1) for child in children]
        leaf = sum(downward_leaves[1:], start=downward_leaves[0])

    merge_level = profile.levels - level
        
    for axis, forest in enumerate(factor_forests):
        U_grid = make_grid(profile, forest, axis, 1, up=True, merge_level = merge_level)
        leaf.axis_multiply(U_grid, axis, True)

    return leaf

def apply(profile, A, factor_forests):
    assert len(factor_forests) == profile.dimens
    split = 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels)
    A_grid = TensorGrid(A, split)
    base_grid = deepcopy(A_grid)

    base_dict = apply_down(profile, A_grid, factor_forests, profile.levels)

    if profile.direction == DOWN:
        def matrix_at(down_position, up_slice):
            down_up_rows_lists, down_mat_grid = base_dict[tuple(down_position)]
            down_mat = down_mat_grid.get((0, 0))
            down_rows_lists = [du((0, 0))[0] for du in down_up_rows_lists]
            return AK(profile, down_mat, down_rows_lists + up_slice)
        cols_list = np.array_split(range(profile.N), split)
        def block(position, up_slice):
            if len(position) == profile.dimens:
                return matrix_at(position, up_slice)
            return [block(position + [i], up_slice + [cols]) for i, cols in enumerate(cols_list)]
        return np.block(block([], []))

    def matrix_at(down_position, up_position):
        down_up_rows_lists, down_mat_grid = base_dict[tuple(down_position)]
        down_rows_lists = [du(up_position)[0] for du in down_up_rows_lists]
        down_up_rows_lists, up_mat_grid = base_dict[tuple(up_position)]
        up_rows_lists = [du(down_position)[1] for du in down_up_rows_lists]
        return AK(profile, down_mat_grid.get(up_position), down_rows_lists + up_rows_lists)

    def build_upward(up_position, down_position):
        if len(down_position) == profile.dimens:
            return {tuple(down_position): matrix_at(down_position, up_position)}
        up_dict = {}
        for i in range(2 ** profile.levels):
            up_dict.update(build_upward(up_position, down_position + [i]))
        return up_dict

    def get_leaf_grid(up_position):
        up_dict = build_upward(up_position, [])
        grid = deepcopy(base_grid)
        grid.fill_from_positions_dict(up_dict, 2 ** profile.levels, profile.dimens)
        return grid

    return np.block(apply_up(profile, get_leaf_grid, factor_forests, 0).split_A)

    

    
