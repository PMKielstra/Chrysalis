import numpy as np
from scipy.linalg import block_diag
import scipy.linalg.interpolative as interp
from SliceManagement import Multirange, SliceTree
from tensorly import unfold
import itertools
import random
import matplotlib.pyplot as plt

N = 32
eps = 1e-6

def czip(*ls):
    """Zip, but assert that all lists involved have the same length."""
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

def AK(A, coords_list):
    return np.tensordot(A, K_from_coords(coords_list), axes=2)

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
    """A tree that holds a single factorization dimension, up, down, or both."""
    def __init__(self, rows_mats_down, rows_mats_up, position, root):
        self.rows_mats_down = rows_mats_down
        self.rows_mats_up = rows_mats_up
        self.position = position
        self.root = root
        self.children = []

FACTOR_AXIS_SOURCE = 0
FACTOR_AXIS_OBSERVER = 2

def one_level_factor(rows_lists, off_cols, passive_multirange, is_source):
    """Carry out a single step of factorization for a single node in the factor tree.  Treat passive_multirange as either the range for the observer or for the source, depending on the value of is_source."""
    if is_source:
        factor_ranges = [[r, off_cols] + list(passive_multirange) for r in rows_lists]
    else:
        factor_ranges = [list(passive_multirange) + [r, off_cols] for r in rows_lists]
    
    rows_mats = [ss_row_id(fr, FACTOR_AXIS_SOURCE if is_source else FACTOR_AXIS_OBSERVER) for fr in factor_ranges]
    
    if len(rows_mats) > 1:
        new_rows = [np.concatenate((p[0], q[0])) for p, q in czip(rows_mats[::2], rows_mats[1::2])]
    else:
        new_rows = rows_mats[0][0]
    
    return rows_mats, new_rows

BOTH = 0
DOWN = 1
UP = -1

def factor_to_tree(rows_lists_source, rows_lists_observer, off_cols, passive_multirange, level, direction, root=True):
    """Recursively create a factor tree which goes either down, up, or both.  Decrements level every step and stops when it hits zero, allowing for partial factorizations."""
    rows_mats_source, new_rows_source = (None, None) if direction == UP else one_level_factor(rows_lists_source, off_cols, passive_multirange, is_source=True)
    rows_mats_observer, new_rows_observer = (None, None) if direction == DOWN else one_level_factor(rows_lists_observer, off_cols, passive_multirange, is_source=False)
    
    tree = FactorTree(rows_mats_source, rows_mats_observer, passive_multirange.position(), root)
    
    if level > 0:
        tree.children = [factor_to_tree(new_rows_source, new_rows_observer, off_cols, next_step, level - 1, direction, False) for i, next_step in enumerate(passive_multirange.next_steps())]
    
    return tree

def apply_down(split_A, tree):
    """Apply a split matrix down a factor tree, generating a dict of positions of leaves and the resultant split matrix at each one."""
    A_rows_mats_down = []
    
    if tree.root: # To avoid having to split the matrix one more time than necessary, we treat the root node specially.
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
    else:
        for child in tree.children:
            positions_dict.update(apply_down([Arm[1] for Arm in A_rows_mats_down], child))

    return positions_dict

def apply_up(get_transposed_split_As, tree):
    """Apply a split matrix up a factor tree, generating a matrix.  Takes a function which finds the value of the matrices at a given leaf of the tree.

    Awkwardly, this function still takes the root of the tree and recurses downward, but it's a head recursion.  The apply_down function's recursive call is more toward the tail."""
    if tree.children == []:
        return np.concatenate(get_transposed_split_As(tree), axis=0)
    split_As = [apply_up(get_transposed_split_As, child) for child in tree.children]
    return block_diag(*([rm[1] for rm in tree.rows_mats_up])).dot(sum(split_As))

def build_factor_forest(levels, direction=BOTH):
    """Build a "factor forest": a tuple (levels, off_cols_lists, trees, direction), giving the number of levels of factorization, the lists of columns for each tree, the list of actual factor trees, and the direction of factorization (UP, DOWN, or BOTH)."""
    off_split_number = 2 ** levels if direction == BOTH else 1
    off_cols_lists = np.array_split(range(N), off_split_number)
    rows_list = np.array_split(range(N), 2 ** (2 * levels if direction == BOTH else levels))
    passive = Multirange([SliceTree(list(range(N))), SliceTree(list(range(N)))], [2, 2])
    
    trees = [factor_to_tree(rows_list, rows_list, off_cols, passive, levels, direction) for off_cols in off_cols_lists]
        
    return levels, off_cols_lists, trees, direction

def apply(A, factor_forest):
    """Carry out a matrix-tensor product using the factor forest."""
    MSG_APPLYING_DOWN = "Applying down..."
    MSG_APPLYING_UP = "Applying up..."
    
    levels, off_cols_lists, trees, direction = factor_forest

    # Step 1: Apply A down the factor forest, unless we're only applying up.
    if direction in (BOTH, DOWN):
        print(MSG_APPLYING_DOWN)
        transpose_dicts = [apply_down(np.split(A[:, cols], 2 ** (2 * levels if direction == BOTH else levels)), tree) for cols, tree in czip(off_cols_lists, trees)]

    # Step 2: If necessary, split up the column indices into equal chunks (which can do double duty as equal chunks of row indices).
    if direction in (DOWN, UP):
        off_cols_lists = np.array_split(off_cols_lists[0], 2 ** levels)

    # Step 3: Reassemble the split matrix, apply up the factor forest, or both.
    if direction == DOWN:
        # We've already done all the matrix compressions we're going to do; the only thing left is to compute the actual matrix-tensor product and reassemble the block matrix.
        
        assert len(transpose_dicts) == 1
        
        block = []
        for x in range(2 ** levels):
            row = []
            for y in range(2 ** levels):
                down_rows, down_mat = transpose_dicts[0][(x, y)][0]
                down_cols = list(range(N))
                up_rows, up_cols = off_cols_lists[x], off_cols_lists[y]
                row.append(AK(down_mat, (down_rows, down_cols, up_rows, up_cols)))
            block.append(row)
        return np.block(block)
    
    elif direction == UP:
        # We haven't done any matrix compressions yet, so we just split A into even-sized chunks and apply it up the forest.
        
        def get_KA_leaf(up_leaf):
            
            assert len(up_leaf.rows_mats_up) == 1
            
            x, y = up_leaf.position
            down_rows, down_cols = off_cols_lists[x], off_cols_lists[y]
            up_cols = list(range(N))
            up_rows, up_mat = up_leaf.rows_mats_up[0]
            
            return [up_mat.dot(AK(A[down_rows][:, down_cols], (down_rows, down_cols, up_rows, up_cols)))]
        
        print(MSG_APPLYING_UP)
        split_KA = apply_up(get_KA_leaf, trees[0])
        return split_KA
    
    elif direction == BOTH:
        # We've already done some compressions in the downward direction, and we'll need to do some more in the upward direction.  At the center, we carry out a transposition.
        
        def get_transposed_KA_leaf(col_split_position, up_leaf):
            
            assert len(up_leaf.rows_mats_up) == 2 ** levels
            
            x, y = up_leaf.position
            up_cols = off_cols_lists[col_split_position]
            down_cols = off_cols_lists[y]
            
            split_As = []
            for w in range(2 ** levels):
                up_rows, up_mat = up_leaf.rows_mats_up[w]
                down_rows, down_mat = transpose_dicts[y][(w, col_split_position)][x]
                split_As.append(up_mat.dot(AK(down_mat, (down_rows, down_cols, up_rows, up_cols))))
            return split_As
        
        print(MSG_APPLYING_UP)
        split_KA = [apply_up(lambda leaf: get_transposed_KA_leaf(i, leaf), tree) for i, tree in enumerate(trees)]
        return np.concatenate(split_KA, axis=1)
    
    raise Exception(f"{direction} is not a valid direction!")

A = np.random.rand(N, N)
factor_forest = build_factor_forest(2, BOTH)
compressed_A = apply(A, factor_forest)
true_A = np.tensordot(A, K_from_coords([list(range(N)), list(range(N)), list(range(N)), list(range(N))]), axes=2)
print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(true_A))
