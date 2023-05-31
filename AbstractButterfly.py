from math import floor, ceil, log2

from ListTranspose import list_transpose, push_front_to_back

def tree_depth(bf, A, min_leaf_size, axes):
    return floor(log2(floor(min([bf.shape(A, axis) / min_leaf_size for axis in axes]))))

def __butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, steps, depth):
    """Carry out a one-dimensional butterfly factorization along factor_axis, splitting along aux_axis."""

    def merge_list_doubles(l):
        return [bf.stack(l[2*i:2*i+2], axis=factor_axis) for i in range(len(l) // 2)]

    # Step 0: General setup
    factorization = bf.compose(None, None, None)
    singular_values_left = (factor_axis > aux_axis)

    # Step 1: Factorization at leaf nodes
    leaves = bf.split(A, factor_axis, 2 ** depth)
    factored_leaves = list_transpose([bf.factor(leaf, factor_axis, aux_axis) for leaf in leaves], 0, 1)
    Us, Es = (factored_leaves[1], factored_leaves[0]) if singular_values_left else (factored_leaves[0], factored_leaves[1])
    factorization = bf.compose(factorization, bf.diag(Us), factor_axis) # Shortcut the U assembly
    
    # Step 2: Setup for iteration
    E_blocks = [Es] # E = diag(map(merge, E_blocks))
    U_blocks = [Us]

    # Step 3: Process a single E block and a single U block
    def Es_to_Es_and_Rs(Es):
        split_Es = list_transpose([bf.split(E, aux_axis, 2) for E in merge_list_doubles(Es)], 0, 1)
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

    # Step 4: Process all the blocks
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

    # Step 5: Party!
    return factorization_with_head, factorization, U_blocks

def single_axis_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, steps=None):
    """Carry out a butterfly factorization along one axis against another."""
    depth = tree_depth(bf, A, min_leaf_size, [factor_axis, aux_axis])
    if steps == None:
        steps = depth
    full_factorization, _a, _b = __butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, min(steps, depth), depth)
    return full_factorization

def multi_axis_butterfly(bf, A, min_leaf_size, axis_pairs, steps_per_axis=None):
    """Carry out a butterfly factorization along some number of axes.  Takes a list of axis pairs of the form (factor_axis, aux_axis).  This function always calculates a center block from scratch, so, if you know you only want to use one axis, it's more efficient to use single_axis_butterfly."""
    
    dimens = len(axis_pairs)

    # Step 0: Determine number of factorization steps for each pair of axes
    if steps_per_axis == None:
        steps_per_axis = [0] * dimens
    else:
        assert dimens == len(axis_pairs)

    def auto_steps_depth(factor_axis, aux_axis, given_steps):
        depth = tree_depth(bf, A, min_leaf_size, [factor_axis, aux_axis])
        # TODO: This should do proper n-cycle detection.
        if (aux_axis, factor_axis) in axis_pairs: # If the pair (a, b) also appears as (b, a), only factor halfway.  They'll meet in the middle.
            if factor_axis < aux_axis: # Disambiguation: the "smaller" axis, according to Pythonic ordering, gets the smaller number of steps.  This is completely arbitrary.
                steps = floor(depth / 2)
            else:
                steps = ceil(depth / 2)
        else:
            steps = depth
        return steps if given_steps < 1 else min(steps, given_steps), depth
    
    steps_depths = [auto_steps_depth(*axes, given_steps) for axes, given_steps in zip(axis_pairs, steps_per_axis)] 

    # Step 1: Do individual one-dimensional factorizations
    def factorization_and_blocks(i):
        _, factorization, blocks = __butterfly(bf, A, min_leaf_size, axis_pairs[i][0], axis_pairs[i][1], steps_depths[i][0], steps_depths[i][1])
        return factorization, blocks

    factorizations_and_blocks = [factorization_and_blocks(i) for i in range(dimens)]

    blocks = [fb[1] for fb in factorizations_and_blocks]

    # Step 2: Build central block
    def recursive_split_and_build(i, positions, A):
        # Recursion phase 1: split along each auxiliary axis.  (NOTE: You might expect to split along the factor axis, but, in fact, by the end of a single-axis factorization, we are at the other end of the butterfly tree and the roles of the factor and auxiliary axes have fully switched.)
        # Keep track of position along each axis during the split.
        if i < dimens:
            return [
                    recursive_split_and_build(i + 1, {**positions, axis_pairs[i][1]: j}, AA) for j, AA in enumerate(bf.split(A, axis_pairs[i][1], len(blocks[i])))
                ]
        # Recursion phase 2: for each dimension, get the relevant block and call build_center.
        for d in range(dimens):
            U = blocks[d][positions.get(axis_pairs[d][1], None)][positions.get(axis_pairs[d][0], 0)] # First position should always exist -- we've split the original object that way -- so use a default of None to raise an error if it doesn't.  Second position will only exist if the axis pair in question appears both ways ((a, b) and (b, a)), for a two-axis factorization, so just use 0 if it doesn't exist.  (In that case, the list in question should have only one element anyway.)
            A = bf.build_center(A, bf.transpose(U, *axis_pairs[d]), axis_pairs[d][0])
        return A

    central_split = recursive_split_and_build(0, {}, A)
    central = bf.diag(central_split, dimens)

    # Step 3: Combine into one factorization.
    final = bf.compose(factorizations_and_blocks[0][0], central, axis_pairs[0][0])
    for i in range(1, dimens):
        final = bf.join(final, factorizations_and_blocks[i][0])

    # Step 4: Party!
    return final

    
    
