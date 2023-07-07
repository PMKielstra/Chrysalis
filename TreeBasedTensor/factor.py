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

    def __init__(self, N, dimens, eps, levels, direction=BOTH, factor_source=None, factor_observer=None):
        assert N > 0
        assert dimens > 0
        assert eps < 1
        assert direction in (BOTH, UP, DOWN)
        
        if factor_source == None:
            factor_source = 0
            
        if factor_observer == None:
            factor_observer = dimens

        self.N = N
        self.dimens = dimens
        self.eps = eps
        self.levels = levels
        self.direction = direction
        self.factor_source = factor_source
        self.factor_observer = factor_observer
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
            subsamples.append(subsample(sr))
        else:
            subsamples.append(sr)
    
    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = K_from_coords(profile.N, subsamples)

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
    def __init__(self, rows_mats_down, rows_mats_up, position, root):
        self.rows_mats_down = rows_mats_down
        self.rows_mats_up = rows_mats_up
        self.position = position
        self.root = root
        self.children = []

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

def factor_to_tree(profile, rows_lists_source, rows_lists_observer, off_cols, passive_multirange, level, root=True):
    """Recursively create a factor tree which goes either down, up, or both.  Decrements level every step and stops when it hits zero, allowing for partial factorizations."""
    rows_mats_source, new_rows_source = (None, None) if profile.direction == UP else one_level_factor(profile, rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = (None, None) if profile.direction == DOWN else one_level_factor(profile, rows_lists_observer, off_cols, passive_multirange, is_source=False)
    
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange.position(), root)
    
    if level > 0:
        tree.children = [factor_to_tree(profile, new_rows_source, new_rows_observer, off_cols, next_step, level - 1, False) for i, next_step in enumerate(passive_multirange.next_steps())]
    
    return tree

def build_factor_forest(profile):
    """Build a "factor forest": a tuple (levels, off_cols_lists, trees, direction), giving the number of levels of factorization, the lists of columns for each tree, the list of actual factor trees, and the direction of factorization (UP, DOWN, or BOTH)."""

    off_cols_lists = np.array_split(range(profile.N), profile.off_split_number)
    rows_list = np.array_split(range(profile.N), 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels))
    passive = Multirange([SliceTree(list(range(profile.N))) for _ in range(profile.dimens)], [2 for _ in range(profile.dimens)])

    def make_trees(previous_off_cols, level):
        if level == 0:
            return factor_to_tree(profile, rows_list, rows_list, previous_off_cols, passive, profile.levels)
        return [make_trees(previous_off_cols + [off_cols], level - 1) for off_cols in off_cols_lists]
            
    return off_cols_lists, make_trees([], profile.dimens - 1)
