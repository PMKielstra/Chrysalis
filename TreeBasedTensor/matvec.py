import numpy as np
import scipy as sp

from czip import czip
from factor import BOTH, UP, DOWN
from tensor import AK

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
    return sp.linalg.block_diag(*([rm[1] for rm in tree.rows_mats_up])).dot(sum(split_As))

def apply(A, factor_forest):
    """Carry out a matrix-tensor product using the factor forest."""
    MSG_APPLYING_DOWN = "Applying down..."
    MSG_APPLYING_UP = "Applying up..."
    
    N, levels, off_cols_lists, trees, direction = factor_forest

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
                row.append(AK(down_mat, N, (down_rows, down_cols, up_rows, up_cols)))
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
            
            return [up_mat.dot(AK(A[down_rows][:, down_cols], N, (down_rows, down_cols, up_rows, up_cols)))]
        
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
                split_As.append(up_mat.dot(AK(down_mat, N, (down_rows, down_cols, up_rows, up_cols))))
            return split_As
        
        print(MSG_APPLYING_UP)
        split_KA = [apply_up(lambda leaf: get_transposed_KA_leaf(i, leaf), tree) for i, tree in enumerate(trees)]
        return np.concatenate(split_KA, axis=1)
    
    raise Exception(f"{direction} is not a valid direction!")
