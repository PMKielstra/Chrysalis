import numpy as np
from factor import build_factor_forest, UP, DOWN, BOTH
from matvec import apply
from tensor import AK_true
from profiling import ss_accuracy, total_memory

N = 32
eps = 1e-6

A = np.random.rand(N, N)
factor_forest = build_factor_forest(N, eps, 2, BOTH)
print(total_memory(factor_forest))
compressed_AK = apply(A, factor_forest)
print(ss_accuracy(A, compressed_AK, N))
