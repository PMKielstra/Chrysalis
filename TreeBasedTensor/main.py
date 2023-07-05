import numpy as np
from factor import build_factor_forest, UP, DOWN, BOTH
from matvec import apply
from tensor import AK_true

N = 32
eps = 1e-6

A = np.random.rand(N, N)
factor_forest = build_factor_forest(N, eps, 2, BOTH)
compressed_A = apply(A, factor_forest)
true_A = AK_true(A, N)
print(np.linalg.norm(compressed_A - true_A) / np.linalg.norm(true_A))
