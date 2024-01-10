from copy import deepcopy
import itertools
from math import floor

import numpy as np
import scipy as sp

from utils import without, czip, slice_by_index, multilevel_access
from factorprofile import BOTH, UP, DOWN
from tensor import K_from_coords
from tensorgrid import TensorGrid

def make_grid(profile, factor_forest, axis, off_multiplication, up=False, merge_level=0):
    def get_U_at(position):
        position = list(position)
        off_position = without(position, axis)
        off_position = [p // off_multiplication for p in off_position]
        factor_tree = multilevel_access(factor_forest, off_position)
        return factor_tree.rows_mats_down[position[axis]][0], factor_tree.rows_mats_up[position[axis]][0]
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
    if test_tree.children == []:
        return {test_tree.position: ([make_grid(profile, forest, axis, 2 ** level) for axis, forest in enumerate(factor_forests)], deepcopy(A_grid))}

    base_dict = {}
    for c in forest_children(profile, factor_forests):
        base_dict.update(apply_down(profile, deepcopy(A_grid), c, level - 1))
    return base_dict

def ragged_pad(vecs):
    length = max(len(v) for v in vecs)
    return np.array([np.pad(v, (0, length - len(v))) for v in vecs])

def compressed_core_rank(profile, factor_forests):
    assert len(factor_forests) == profile.dimens
    split = 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels)

    base_dict = apply_down(profile, None, factor_forests, profile.levels)

    def coords_at(down_position, up_position):
        down_up_rows_lists, down_mat_grid = base_dict[tuple(down_position)]
        down_rows_lists = [du(up_position)[0] for du in down_up_rows_lists]
        down_up_rows_lists, up_mat_grid = base_dict[tuple(up_position)]
        up_rows_lists = [du(down_position)[1] for du in down_up_rows_lists]
        
        return down_rows_lists + up_rows_lists

    offset = profile.N // (2 ** profile.levels)

    def union_wrt_offset(l, r):
        while floor(l[0] / offset) < floor(r[0] / offset):
            r = r - offset
        while floor(l[0] / offset) > floor(r[0] / offset):
            r = r + offset
        return np.union1d(l, r)

    all_down_positions = itertools.product(*(range(2 ** profile.levels) for _ in range(profile.dimens)))
    coords = [coords_at(dp, dp) for dp in all_down_positions]
    for l in range(len(coords[0])):
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i == j:
                    continue
                coords[i][l] = union_wrt_offset(coords[i][l], coords[j][l])
        coords[i][l] = np.sort(coords[i][l])
    vecs = [K_from_coords(profile, c).ravel() for c in coords]
    storage_matrix = ragged_pad(vecs).T
    return np.linalg.matrix_rank(storage_matrix, tol=profile.eps), storage_matrix.shape
    

    
