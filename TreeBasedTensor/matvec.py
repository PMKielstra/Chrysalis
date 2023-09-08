import numpy as np
import scipy as sp

from utils import czip, slice_by_index, multilevel_access
from factorprofile import BOTH, UP, DOWN
from tensor import AK

def apply_down(split_A, tree):
    """Apply a split matrix down a factor tree, generating a dict of positions of leaves and the resultant split matrix at each one."""
    A_rows_mats_down = []
    
    if tree.root: # To avoid having to split the matrix one more time than necessary, we treat the root node specially.
        for row_mat, A in czip(tree.rows_mats_down, split_A):
            new_A = np.tensordot(row_mat[1].T, A, axes=1)
            A_rows_mats_down.append((row_mat[0], new_A))
    else:
        for row_mat, A_1, A_2 in czip(tree.rows_mats_down, split_A[::2], split_A[1::2]):
            new_A = np.tensordot(row_mat[1].T, np.concatenate((A_1, A_2)), axes=1)
            A_rows_mats_down.append((row_mat[0], new_A))
            
    positions_dict = {}
    if tree.children == []:
        positions_dict[tree.position] = A_rows_mats_down
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
    return np.tensordot(sp.linalg.block_diag(*([rm[1] for rm in tree.rows_mats_up])), sum(split_As), axes=1)

def apply(profile, A, factor_forest):
    """Carry out a matrix-tensor product using the factor forest."""
    MSG_APPLYING_DOWN = "Applying down..."
    MSG_APPLYING_UP = "Applying up..."
    
    off_cols_lists, trees = factor_forest

    if profile.as_matrix:
        A = np.ravel(A)

    # Step 1: Apply A down the factor forest, unless we're only applying up.
    if profile.direction in (BOTH, DOWN):
        if profile.verbose:
            print(MSG_APPLYING_DOWN)
        def build_transpose_dict(A, trees, i):
            if i == profile.dimens:
                return apply_down(np.split(A, 2 ** (2 * profile.levels if profile.direction == BOTH else profile.levels)), trees) # At this level, "trees" should in fact be a single tree.
            return [build_transpose_dict(A[tuple([slice(None)] * i + [cols])], tree, i + 1) for cols, tree in czip(off_cols_lists, trees)]
        transpose_dicts = build_transpose_dict(A, trees, 1)

    # Step 2: If necessary, split up the column indices into equal chunks (which can do double duty as equal chunks of row indices).
    if profile.direction in (DOWN, UP):
        off_cols_lists = np.array_split(off_cols_lists[0], 2 ** profile.levels)

    # Step 3: Reassemble the split matrix, apply up the factor forest, or both.
    if profile.direction == DOWN:
        # We've already done all the matrix compressions we're going to do; the only thing left is to compute the actual matrix-tensor product and reassemble the block matrix.
        
        def block(dimen, position):
            if dimen == 0:
                down_rows, down_mat = multilevel_access(transpose_dicts, [0] * (profile.dimens - 1), assert_single_element=True)[tuple(position)][0]
                return AK(profile, down_mat, [down_rows] + [list(range(profile.N))] * (profile.dimens - 1) + [off_cols_lists[i] for i in position])
            return [block(dimen - 1, position + [i]) for i in range(2 ** profile.levels)]
        return np.block(block(profile.dimens, []))
    
    elif profile.direction == UP:
        # We haven't done any matrix compressions yet, so we just split A into even-sized chunks and apply it up the forest.
        
        def get_KA_leaf(up_leaf):
            assert len(up_leaf.rows_mats_up) == 1
            
            down_index = [off_cols_lists[i] for i in up_leaf.position]
            
            up_rows, up_mat = up_leaf.rows_mats_up[0]
            return [np.tensordot(up_mat, AK(profile, slice_by_index(A, down_index), down_index + [up_rows] + [list(range(profile.N))] * (profile.dimens - 1)), axes=1)]

        if profile.verbose:
            print(MSG_APPLYING_UP)
        split_KA = apply_up(get_KA_leaf, multilevel_access(trees, [0] * (profile.dimens - 1), assert_single_element=True))
        return split_KA
    
    elif profile.direction == BOTH:
        # We've already done some compressions in the downward direction, and we'll need to do some more in the upward direction.  At the center, we carry out a transposition.
        
        def get_transposed_KA_leaf(col_split_position, up_leaf):

            assert len(up_leaf.rows_mats_up) == 2 ** profile.levels
            
            up_cols = [off_cols_lists[j] for j in col_split_position]
            down_cols = [off_cols_lists[i] for i in up_leaf.position[1:]]
            
            split_As = []
            for w in range(2 ** profile.levels):
                up_rows, up_mat = up_leaf.rows_mats_up[w]
                down_rows, down_mat = multilevel_access(transpose_dicts, up_leaf.position[1:])[tuple([w] + col_split_position)][up_leaf.position[0]]
                split_As.append(np.tensordot(up_mat, AK(profile, down_mat, [down_rows] + down_cols + [up_rows] + up_cols), axes=1))
            return split_As

        if profile.verbose:
            print(MSG_APPLYING_UP)

        def apply_and_join(axis, col_split_position, trees):
            if axis == profile.dimens:
                return apply_up(lambda leaf: get_transposed_KA_leaf(col_split_position, leaf), trees) # At this level, "trees" should in fact be a single tree.
            return np.concatenate([apply_and_join(axis + 1, col_split_position + [i], tree) for i, tree in enumerate(trees)], axis=axis)
        
        return apply_and_join(1, [], trees)
    
    raise Exception(f"{direction} is not a valid direction!")
