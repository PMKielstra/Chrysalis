from concurrent.futures import Future, as_completed
from math import ceil
from itertools import starmap
from functools import reduce
import os
import time
from copy import deepcopy

import numpy as np
import scipy.linalg.interpolative as interp
import tensorly

from utils import czip, subsample
from multirange import SliceTree, Multirange
from tensor import K_from_coords
from profile import BOTH, UP, DOWN

def translate_union(m, n):
    """Reduce n to one element by increasing the size of m and using translation invariance."""
    shift = np.max(n) - np.min(n)
    return reduce(np.union1d, (np.array(m) - i for i in range(shift)))
    
def ss_row_id(profile, sampled_ranges, is_source):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    factor_index = profile.factor_index(is_source)

    # Step 0: Use translation invariance if possible
    if profile.translation_invariant:
        sampled_ranges = deepcopy(sampled_ranges)
        for i in range(1, profile.dimens):
            observer_index = factor_index + i
            source_index = (observer_index + profile.dimens) % (2 * profile.dimens)
            sampled_ranges[source_index] = translate_union(sampled_ranges[source_index], sampled_ranges[observer_index])
            sampled_ranges[observer_index] = sampled_ranges[observer_index][:1]
    
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

def print_level(profile, level, start_time):
    if profile.verbose:
        print(f"Factored level {level} of {profile.levels} ({time.time() - start_time})", flush=True)

def sequential_factor_to_tree(profile, off_cols, rows_lists_source=None, rows_lists_observer=None, passive_multirange=None, level=0, root=True, far_left=True, start_time=0):
    """Recursively create a factor tree which goes either down, up, or both.  Works sequentially."""    
    if rows_lists_source is None:
        rows_lists_source = np.array_split(range(profile.N), 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels))
    if rows_lists_observer is None:
        rows_lists_observer = np.array_split(range(profile.N), 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels))
    if passive_multirange is None:
        passive_multirange = Multirange([SliceTree(list(range(profile.N))) for _ in range(profile.dimens)], [2 for _ in range(profile.dimens)])

    start_time = time.time()
    
    rows_mats_source, new_rows_source = (None, None) if profile.direction == UP else one_level_factor(profile, rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = (None, None) if profile.direction == DOWN else one_level_factor(profile, rows_lists_observer, off_cols, passive_multirange, is_source=False)
    
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange, root)

    print_level(profile, level, start_time)
    
    if level < profile.levels:
        tree.children = [factor_to_tree(profile, off_cols, new_rows_source, new_rows_observer, next_step, level + 1, root=False, far_left=(far_left and i == 0), start_time=start_time) for i, next_step in enumerate(passive_multirange.next_steps())]

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
    for level in range(profile.levels):
        start_time = time.time()
        new_leaves = []
        for leaf in leaves:
            leaf.collect()
            new_rows_source = leaf.new_rows(True)
            new_rows_observer = leaf.new_rows(False)
            child_multiranges = leaf.multirange.next_steps()
            leaf.children = [make_async_tree(pool, profile, off_cols, new_rows_source, new_rows_observer, cm, False, chunksize) for cm in child_multiranges]
            new_leaves += leaf.children
        leaves = new_leaves
        print_level(profile, level, start_time)
    for l in leaves:
        l.collect()
    print_level(profile, profile.levels, start_time)
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
