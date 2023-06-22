import numpy as np
import scipy as sp

class SparseDiag:
    def __init__(self, matrices):
        self.matrices = matrices
        self.shape = np.sum([m.shape for m in matrices], axis=0)
    def __array__(self, dtype=None):
        return sp.linalg.block_diag(*self.matrices)
