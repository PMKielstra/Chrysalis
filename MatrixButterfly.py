import numpy as np
import scipy as sp
from copy import deepcopy

import matplotlib.pyplot as plt

from ButterflyInterface import ButterflyInterface

class MatrixButterfly(ButterflyInterface):
    """A class that implements the methods required for butterfly decomposition of matrices.  Takes a truncation tolerance to be used for cutting off SVDs."""
    
    def __init__(self, truncation_tolerance):
        self.truncation_tolerance = truncation_tolerance
    
    def shape(self, A, axis):
        return A.shape[axis]

    def split(self, A, axis, n):
        return np.array_split(A, n, axis=axis)

    def factor(self, A, singular_values_left):
        U, S, Vh = sp.linalg.svd(A, full_matrices=False, compute_uv=True)
        k = (np.abs(S) > self.truncation_tolerance * np.abs(S[0])).nonzero()[0][-1]
        return (U[:, :k].dot(np.diag(S[:k])), Vh[:k, :]) if singular_values_left else (U[:, :k], np.diag(S[:k]).dot(Vh[:k, :]))

    def merge(self, As, axis=0):
        return np.concatenate(As, axis=axis)
    showing = False
    def recursive_pad(self, us, dimen, dimens):
        if dimens == 1:
            below = sum([u.shape[dimen] for u in us])
            above = 0
            new_us = []
            for u in us:
                padding_sequence = [(0, 0)] * 2
                below -= u.shape[dimen]
                padding_sequence[dimen] = (above, below)
                above += u.shape[dimen]
                new_us.append(np.pad(u, padding_sequence))
                if self.showing:
                    plt.spy(new_us[-1])
                    plt.show()
            return new_us
        else:
            #self.showing = True
            return [self.recursive_pad(u, dimen, dimens - 1) for u in us]

    def diag(self, us, dimens=1):
        if dimens == 1:
            return sp.linalg.block_diag(*us)
        us = self.transpose(us, 0, dimens - 1)
        for dimen in range(dimens):
            us = self.recursive_pad(us, dimens - 1 - dimen, dimens)
            us = self.transpose(us, 0, dimens - 1)
        axes_list = [1] if dimens==1 else [1, 0]
        return self.recursive_merge(us, axes_list)
    
    def compose(self, L, A, compose_left):
        if L == None:
            return []
        else:
            return ([A] + L) if compose_left else (L + [A])

    def join(self, L1, L2):
        return L1 + L2

    def transpose(self, A, ax0, ax1):
        return np.swapaxes(A, ax0, ax1)

    def contract(self, As):
        As = deepcopy(As) # Avoid accidentally altering the As list
        X = As.pop()
        while len(As) > 0:
            X = As.pop().dot(X)
        return X
