import numpy as np
import scipy as sp
import scipy.linalg.interpolative as interp
from copy import deepcopy

from ListTranspose import list_transpose
from ButterflyInterface import ButterflyInterface

class TensorButterfly(ButterflyInterface):
    """A class that implements the methods required for butterfly decomposition of 4-tensors.  Takes a truncation tolerance to be used for cutting off IDs."""
    
    def __init__(self, truncation_tolerance, tensor_dimension = 4):
        self.truncation_tolerance = truncation_tolerance
        self.tensor_dimension = tensor_dimension
    
    def shape(self, A, axis):
        return A.shape[axis]

    def multiply(self, A, B, axis=0):
        return np.tensordot(A, B, ((axis,), (axis,)))

    def split(self, A, axis, n):
        return np.array_split(A, n, axis=axis)

    def matrix_apply(self, f, X, row_index):
        cols = np.prod(X.shape) // X.shape[row_index]
        Y = np.reshape(np.swapaxes(X, row_index, -1), (cols, X.shape[row_index]))
        Y, Z = f(Y)
        new_shape = list(X.shape)
        new_shape[row_index], new_shape[-1] = new_shape[-1], Y.shape[-1]
        Y = np.swapaxes(np.reshape(Y, new_shape), row_index, -1)
        return Y, Z

    def factor(self, A, factor_axis, aux_axis):
        singular_values_left = (factor_axis > aux_axis)
        def matrix_ID(A):
            if singular_values_left:
                k, idx, proj = interp.interp_decomp(A, self.truncation_tolerance) # A = (Skeleton)(Interpolation), where Skeleton has the singular values.
                return interp.reconstruct_skel_matrix(A, k, idx), interp.reconstruct_interp_matrix(idx, proj)
            else:
                k, idx, proj = interp.interp_decomp(A.T, self.truncation_tolerance)
                return interp.reconstruct_interp_matrix(idx, proj).T, interp.reconstruct_skel_matrix(A.T, k, idx).T
        return self.matrix_apply(matrix_ID, A, factor_axis) 
        

    def find_identity_row(self, U, r):
        eps = 1e-15
        comparison_row = np.zeros_like(U[0])
        comparison_row[r] = 1
        for i in range(len(U)):
            if np.linalg.norm(U[i] - comparison_row) < eps:
                return i
        raise Exception("Could not find identity row.")

    def build_center(self, K, U, axis):
        if self.decomposition == 'svd':
            return U.dot(K) if axis == 0 else K.dot(U)
        elif self.decomposition == 'id':
            if axis == 0: # This is easier than writing the code for a general axis
                U = U.T
            else:
                K = K.T
            rows = []
            for r in range(U.shape[1]):
                rows.append(K[self.find_identity_row(U, r)])
            final = np.array(rows)
            return final.T if axis == 1 else final
        return None

    def stack(self, As, axis=0):
        return np.concatenate(As, axis=axis)

    def recursive_pad(self, us, dimen, dimens):
        if dimens == 1:
            below = sum([u.shape[dimen] for u in us])
            above = 0
            new_us = []
            for u in us:
                padding_sequence = [(0, 0)] * len(us[0].shape)
                below -= u.shape[dimen]
                padding_sequence[dimen] = (above, below)
                above += u.shape[dimen]
                new_us.append(np.pad(u, padding_sequence))
            return new_us
        else:
            return [self.recursive_pad(u, dimen, dimens - 1) for u in us]

    def diag(self, us, dimens=1):
        tensor_dim = len(us[0].shape)
        if dimens == 1 and False:
            return sp.linalg.block_diag(*us)
        us = list_transpose(us, 0, dimens - 1)
        for dimen in range(dimens):
            us = self.recursive_pad(us, dimens - 1 - dimen, dimens)
            us = list_transpose(us, 0, dimens - 1)
        axes_list = list(range(tensor_dim - 1, tensor_dim - 1 - dimens, -1))
        return self.recursive_stack(us, axes_list)

    def transpose(self, A, ax0, ax1):
        return np.conjugate(np.swapaxes(A, ax0, ax1))
