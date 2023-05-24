### PART 1: AN INTERFACE FOR FACTORIZABLE THINGS

from abc import ABC, abstractmethod

class ButterflyInterface(ABC):
    """The most basic operation set required to carry out a butterfly factorization.  All docstrings are written as if we were factorizing matrices, but this should in theory work for matrices, tensors, bananas, whatever."""
    
    @abstractmethod
    def shape(self, A, axis):
        """The size of the matrix A along the given axis."""
        pass
    
    @abstractmethod
    def split(self, A, axis, n):
        """Split A into n parts along the given axis."""
        pass

    @abstractmethod
    def factor(self, A, singular_values_left):
        """Truncated SVD factorization for A.  Return (U.S, V^T) if singular_values_left, otherwise (U, S.V^T)."""
        pass

    @abstractmethod
    def diag(self, us):
        """The diagonal matrix [[us[0], 0, 0, ...], [0, us[1], 0, ...], [0, 0, us[2], ...], ...]."""
        pass

    @abstractmethod
    def merge(self, As, axis=0):
        """Stack the As along the given axis."""
        pass

    @abstractmethod
    def compose(self, L, A, compose_left):
        """[A] + L, L + [A], or [] if L is None."""
        pass

    @abstractmethod
    def join(self, L1, L2):
        """Blindly join factorizations"""
        pass

    @abstractmethod
    def transpose(self, As, ax0, ax1):
        """Swap ax0 and ax1 in each element of the composed As."""
        pass

    @abstractmethod
    def contract(self, As):
        """Contract the As (the result of calling compose) to a single matrix."""
        pass

    def apply(self, As, X):
        """Freebie: apply the As to a particular X."""
        return self.contract(self.compose(As, X, False))

### PART 2: ABSTRACT FACTORIZATION

from math import floor, log2
from copy import deepcopy

def tree_depth(bf, A, min_leaf_size, axes):
    return floor(log2(floor(min([bf.shape(A, axis) / min_leaf_size for axis in axes]))))

def single_axis_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, steps, depth):
    """Carry out a one-dimensional butterfly factorization along factor_axis, splitting along aux_axis."""

    def list_transpose(l):
        return list(map(list, zip(*l)))

    def merge_list_doubles(l):
        return [bf.merge(l[2*i:2*i+2], axis=factor_axis) for i in range(len(l) // 2)]

    # Step 0: general setup
    factorization = bf.compose(None, None, None)
    singular_values_left = (factor_axis > aux_axis)

    # Step 1: factorization at leaf nodes
    leaves = bf.split(A, factor_axis, 2 ** depth)
    factored_leaves = list_transpose([bf.factor(leaf, singular_values_left) for leaf in leaves])
    Us, Es = (factored_leaves[1], factored_leaves[0]) if singular_values_left else (factored_leaves[0], factored_leaves[1])
    factorization = bf.compose(factorization, bf.diag(Us), singular_values_left) # Shortcut the U assembly
    
    # Step 2: setup for iteration
    E_blocks = [Es] # E = diag(map(merge, E_blocks))

    # Step 3: process a single E block
    def Es_to_Es_and_R(Es):
        split_Es = list_transpose([bf.split(E, aux_axis, 2) for E in merge_list_doubles(Es)])
        E_blocks = []
        R_cols = []
        for col in split_Es: # There should be two of these
            R_chunks = []
            E_col = []
            for E in col:
                factored_E = bf.factor(E, singular_values_left)
                R, new_E = (factored_E[1], factored_E[0]) if singular_values_left else (factored_E[0], factored_E[1])
                R_chunks.append(R)
                E_col.append(new_E)
            R_cols.append(bf.diag(R_chunks))
            E_blocks.append(E_col)
        return E_blocks, bf.merge(R_cols, axis=aux_axis)

    # Step 4: process all the blocks
    for _ in range(min(steps, depth)):
        new_E_blocks, Rs = [], []
        for block in E_blocks:
            Es, R = Es_to_Es_and_R(block)
            new_E_blocks += Es
            Rs.append(R)
        E_blocks = new_E_blocks
        factorization = bf.compose(factorization, bf.diag(Rs), singular_values_left)
    final_E_blocks = list(map(lambda E: bf.merge(E, axis=factor_axis), E_blocks))
    final_E = bf.diag(final_E_blocks)
    factorization_with_head = bf.compose(factorization, final_E, singular_values_left)

    # Step 5: party!
    return factorization_with_head, factorization # Also return a "headless" factorization, for use with the 2d butterfly

def one_dimensional_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis):
    depth = tree_depth(bf, A, min_leaf_size, [factor_axis, aux_axis])
    full_factorization, _ = single_axis_butterfly(bf, A, min_leaf_size, factor_axis, aux_axis, depth, depth)
    return full_factorization
from matplotlib import pyplot as plt
def two_dimensional_butterfly(bf, A, min_leaf_size, axes):
    assert len(axes) == 2
    depth = tree_depth(bf, A, min_leaf_size, axes)
    steps = floor(depth / 2)
    _, left_U = single_axis_butterfly(bf, A, min_leaf_size, axes[1], axes[0], steps, depth)
    _, right_V = single_axis_butterfly(bf, A, min_leaf_size, axes[0], axes[1], steps, depth)
    
    left_U_T = bf.transpose(left_U, axes[0], axes[1])
    right_V_T = bf.transpose(right_V, axes[0], axes[1])
    
    center = bf.contract(bf.join(bf.compose(left_U_T, A, False), right_V_T))

    plt.spy(bf.contract(bf.join(right_V_T, right_V)))
    plt.show()
    return bf.join(bf.compose(left_U, center, False), right_V)
