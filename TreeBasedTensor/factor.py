import numpy as np
import scipy.linalg.interpolative as interp
import tensorly

from utils import czip, subsample
from multirange import SliceTree, Multirange
from tensor import K_from_coords

def ss_row_id(N, eps, sampled_ranges, factor_index):
    """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
    # Step 1: Subsample
    subsamples = []
    for i, sr in enumerate(sampled_ranges):
        if i != factor_index:
            subsamples.append(subsample(sr))
        else:
            subsamples.append(sr)
    
    # Step 2: Set up a tensor from the points chosen by the subsampling
    A = K_from_coords(N, subsamples)

    # Step 3: Unfold the tensor and carry out ID
    unfolded = tensorly.unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
    k, idx, proj = interp.interp_decomp(unfolded, eps)
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

FACTOR_AXIS_SOURCE = 0
FACTOR_AXIS_OBSERVER = 2

def one_level_factor(N, eps, rows_lists, off_cols, passive_multirange, is_source):
    """Carry out a single step of factorization for a single node in the factor tree.  Treat passive_multirange as either the range for the observer or for the source, depending on the value of is_source."""
    if is_source:
        factor_ranges = [[r, off_cols] + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r, off_cols] for r in rows_lists]
    
    rows_mats = [ss_row_id(N, eps, fr, FACTOR_AXIS_SOURCE if is_source else FACTOR_AXIS_OBSERVER) for fr in factor_ranges]
    
    if len(rows_mats) > 1:
        new_rows = [np.concatenate((p[0], q[0])) for p, q in czip(rows_mats[::2], rows_mats[1::2])]
    else:
        new_rows = rows_mats[0][0]
    
    return rows_mats, new_rows

BOTH = 0
DOWN = 1
UP = -1

def factor_to_tree(N, eps, rows_lists_source, rows_lists_observer, off_cols, passive_multirange, level, direction, root=True):
    """Recursively create a factor tree which goes either down, up, or both.  Decrements level every step and stops when it hits zero, allowing for partial factorizations."""
    rows_mats_source, new_rows_source = (None, None) if direction == UP else one_level_factor(N, eps, rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = (None, None) if direction == DOWN else one_level_factor(N, eps, rows_lists_observer, off_cols, passive_multirange, is_source=False)
    
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange.position(), root)
    
    if level > 0:
        tree.children = [factor_to_tree(N, eps, new_rows_source, new_rows_observer, off_cols, next_step, level - 1, direction, False) for i, next_step in enumerate(passive_multirange.next_steps())]
    
    return tree

def build_factor_forest(N, eps, levels, direction=BOTH):
    """Build a "factor forest": a tuple (levels, off_cols_lists, trees, direction), giving the number of levels of factorization, the lists of columns for each tree, the list of actual factor trees, and the direction of factorization (UP, DOWN, or BOTH)."""
    assert eps < 1
    assert levels >= 1

    off_split_number = 2 ** levels if direction == BOTH else 1
    off_cols_lists = np.array_split(range(N), off_split_number)
    rows_list = np.array_split(range(N), 2 ** (2 * levels if direction == BOTH else levels))
    passive = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 2])
    
    trees = [factor_to_tree(N, eps, rows_list, rows_list, off_cols, passive, levels, direction) for off_cols in off_cols_lists]
        
    return N, levels, off_cols_lists, trees, direction
