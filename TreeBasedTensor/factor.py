from concurrent.futures import Future, as_completed
from math import ceil
from itertools import starmap
import os

import numpy as np
import scipy.linalg.interpolative as interp
import tensorly

from utils import czip, subsample
from multirange import SliceTree, Multirange
from tensor import K_from_coords

BOTH = 0
DOWN = 1
UP = -1

class Profile:
    """Stores all the parameters necessary for a factorization."""

    def __init__(self, N, dimens, eps, levels, direction=BOTH, subsamples = 20, as_matrix = False, boost_subsamples = True, processes=None):
        assert N > 0
        assert dimens > 0
        assert eps < 1
        assert direction in (BOTH, UP, DOWN)

        self.N = N ** dimens if as_matrix else N
        self.true_N = N
        self.dimens = 1 if as_matrix else dimens
        self.true_dimens = dimens
        self.eps = eps
        self.levels = levels * (dimens if as_matrix else 1)
        self.direction = direction
        self.subsamples = subsamples ** (1 if not as_matrix or not boost_subsamples else dimens)
        self.as_matrix = as_matrix
        self.processes = processes
        self.factor_source = 0 # TODO: add the possibility to factor along different axes.
        self.factor_observer = self.dimens
        self.off_split_number = 2 ** levels if direction == BOTH else 1
        
    def factor_index(self, is_source):
        return self.factor_source if is_source else self.factor_observer
    
def ss_row_id(profile, sampled_ranges, is_source):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    factor_index = profile.factor_index(is_source)
    
    # Step 1: Subsample
    subsamples = []
    for i, sr in enumerate(sampled_ranges):
        if i != factor_index:
            subsamples.append(subsample(sr, profile.subsamples))
        else:
            subsamples.append(sr)
    
    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = K_from_coords(profile, subsamples)

    # Step 3: Unfold the tensor and carry out ID
    unfolded = tensorly.unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, profile.eps)
    R = interp.reconstruct_interp_matrix(idx, proj)

    # Step 4: Map the rows chosen by the ID, which are a subset of [1, ..., len(multirange[factor_index])], back to a subset of the relevant actual rows
    old_rows = sampled_ranges[factor_index]
    new_rows = old_rows[idx[:k]]
    return new_rows, R.T

class FactorTree:
    """A tree that holds a single factorization dimension, up, down, or both."""
    def __init__(self, rows_mats_down_async, rows_mats_up_async, multirange, root):
        self.rows_mats_down = None
        self.rows_mats_up = None
        self.rows_mats_down_async = rows_mats_down_async
        self.rows_mats_up_async = rows_mats_up_async
        self.multirange = multirange
        self.root = root
        self.children = []

    @property
    def position(self):
        return self.multirange.position()

    @property
    def is_async(self):
        return self.rows_mats_down_async is not None or self.rows_mats_up_async is not None

    def collect(self):
        if self.rows_mats_down_async is not None:
            self.rows_mats_down = [q for r in self.rows_mats_down_async for q in r.result()]
        if self.rows_mats_up_async is not None:
            self.rows_mats_up = [q for r in self.rows_mats_up_async for q in r.result()]
        self.rows_mats_down_async, self.rows_mats_up_async = None, None

    def new_rows(self, is_source):
        rows_mats = self.rows_mats_down if is_source else self.rows_mats_up
        if len(rows_mats) > 1:
            new_rows = [np.concatenate((p[0], q[0])) for p, q in czip(rows_mats[::2], rows_mats[1::2])]
        else:
            new_rows = rows_mats[0][0]
        return new_rows

def one_level_factor(profile, rows_lists, off_cols, passive_multirange, is_source):
    """Carry out a single step of factorization for a single node in the factor tree.  Treat passive_multirange as either the range for the observer or for the source, depending on the value of is_source."""
    if is_source:
        factor_ranges = [[r] + off_cols + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r] + off_cols for r in rows_lists]
    
    rows_mats = [ss_row_id(profile, fr, is_source) for fr in factor_ranges]
    
    if len(rows_mats) > 1:
        new_rows = [np.concatenate((p[0], q[0])) for p, q in czip(rows_mats[::2], rows_mats[1::2])]
    else:
        new_rows = rows_mats[0][0]
    
    return rows_mats, new_rows

def sequential_factor_to_tree(profile, off_cols, rows_lists_source=None, rows_lists_observer=None, passive_multirange=None, level=None, root=True):
    """Recursively create a factor tree which goes either down, up, or both.  Decrements level every step and stops when it hits zero, allowing for partial factorizations.  Works sequentially."""
    if rows_lists_source is None:
        rows_lists_source = np.array_split(range(profile.N), 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels))
    if rows_lists_observer is None:
        rows_lists_observer = np.array_split(range(profile.N), 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels))
    if passive_multirange is None:
        passive_multirange = Multirange([SliceTree(list(range(profile.N))) for _ in range(profile.dimens)], [2 for _ in range(profile.dimens)])
    if level is None:
        level = profile.levels
    
    rows_mats_source, new_rows_source = (None, None) if profile.direction == UP else one_level_factor(profile, rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = (None, None) if profile.direction == DOWN else one_level_factor(profile, rows_lists_observer, off_cols, passive_multirange, is_source=False)
    
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange, root)
    
    if level > 0:
        tree.children = [factor_to_tree(profile, off_cols, new_rows_source, new_rows_observer, next_step, level - 1, False) for next_step in passive_multirange.next_steps()]

    return tree

def eager_starmap(func, l):
    return list(starmap(func, l))

def async_map_id(pool, profile, factor_ranges, is_source, chunksize=1):
    results = []
    for i in range(ceil(len(factor_ranges) / chunksize)):
        results.append(pool.submit(eager_starmap, ss_row_id, [(profile, fr, is_source) for fr in factor_ranges[i * chunksize : (i + 1) * chunksize]]))
    return results

def make_async_tree(pool, profile, off_cols, rows_lists_source, rows_lists_observer, multirange, is_root, chunksize):
    """Carry out one step of a factorization, but submit the actual matrices to a thread pool rather than factoring them here and now."""
    factor_ranges_source = [[r] + off_cols + list(multirange) for r in rows_lists_source]
    factor_ranges_observer = [list(multirange) + [r] + off_cols for r in rows_lists_observer]
    return FactorTree(
                async_map_id(pool, profile, factor_ranges_source, True, chunksize),
                async_map_id(pool, profile, factor_ranges_observer, False, chunksize),
                multirange,
                is_root
        )

def parallel_factor_to_tree(pool, profile, off_cols):
    """A parallel equivalent to sequential_factor_to_tree, using a concurrent.futures pool."""
    split_number = 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels)
    rows_lists_source = np.array_split(range(profile.N), split_number)
    rows_lists_observer = np.array_split(range(profile.N), split_number)
    passive_multirange = Multirange([SliceTree(list(range(profile.N))) for _ in range(profile.dimens)], [2 for _ in range(profile.dimens)])
    chunksize = max(split_number // os.cpu_count(), 1) # The number of submatrices to factor on one process at one time.  Since we parallelize across all nodes of any tree at the same level, the total number of factorizations that we care about at any given time should remain constant.
    tree = make_async_tree(pool, profile, off_cols, rows_lists_source, rows_lists_observer, passive_multirange, True, chunksize)
    leaves = [tree]
    for _ in range(profile.levels):
        new_leaves = []
        for l in leaves:
            l.collect()
            new_rows_source = l.new_rows(True)
            new_rows_observer = l.new_rows(False)
            child_multiranges = l.multirange.next_steps()
            l.children = [make_async_tree(pool, profile, off_cols, new_rows_source, new_rows_observer, cm, False, chunksize) for cm in child_multiranges]
            new_leaves += l.children
        leaves = new_leaves
    for l in leaves:
        l.collect()
    return tree

def build_factor_forest(pool, profile):
    """Build a "factor forest": a tuple (levels, off_cols_lists, trees, direction), giving the number of levels of factorization, the lists of columns for each tree, the list of actual factor trees, and the direction of factorization (UP, DOWN, or BOTH)."""

    off_cols_lists = np.array_split(range(profile.N), profile.off_split_number)

    def make_trees(previous_off_cols, level):
        if level == 0:
            if pool is None:
                return sequential_factor_to_tree(profile, previous_off_cols)
            return parallel_factor_to_tree(pool, profile, previous_off_cols)
        return [make_trees(previous_off_cols + [off_cols], level - 1) for off_cols in off_cols_lists]

    return off_cols_lists, make_trees([], profile.dimens - 1)
