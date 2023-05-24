import numpy as np
import scipy as sp
from copy import deepcopy

from AbstractButterfly import ButterflyInterface

class MatrixButterfly(ButterflyInterface):
    """A class that implements the methods required for butterfly decomposition of matrices.  Takes a truncation tolerance to be used for cutting off SVDs."""
    
    def __init__(self, truncation_tolerance):
        self.truncation_tolerance = truncation_tolerance
    
    def shape(self, A, axis):
        return A.shape[axis]

    def split(self, A, axis, n):
        return np.array_split(A, n, axis=axis)

    def factor(self, A, singular_values_left):
        U, S, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        k = (np.abs(S) > self.truncation_tolerance * np.abs(S[0])).nonzero()[0][-1]
        return (U[:, :k].dot(np.diag(S[:k])), Vh[:k, :]) if singular_values_left else (U[:, :k], np.diag(S[:k]).dot(Vh[:k, :]))

    def diag(self, us):
        return sp.linalg.block_diag(*us)

    def merge(self, As, axis=0):
        return np.concatenate(As, axis=axis)

    def compose(self, L, A, compose_left):
        if L == None:
            return []
        else:
            return ([A] + L) if compose_left else (L + [A])

    def join(self, L1, L2):
        return L1 + L2

    def transpose(self, As, ax0, ax1):
        axes = [ax0, ax1]
        axes.sort(reverse=True)
        result = []
        for A in As:
            result.append(np.transpose(A, axes=axes))
        return list(reversed(result))

    def contract(self, As):
        As = deepcopy(As) # Avoid accidentally altering the As list
        X = As.pop()
        while len(As) > 0:
            X = As.pop().dot(X)
        return X
