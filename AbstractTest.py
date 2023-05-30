from AbstractButterfly import one_dimensional_butterfly, two_dimensional_butterfly, multidimensional_butterfly
from MatrixButterfly import MatrixButterfly
import numpy as np
from matplotlib import pyplot as plt

from NoneButterfly import NoneButterfly

def K(r1, r2):
    return np.exp(1j * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def interaction_matrix(N):
    source = np.array([[0, x] for x in np.linspace(0, 1, N)])
    dest = np.array([[1, x] for x in np.linspace(0, 1, N)])
    return np.fromfunction(np.vectorize(lambda i, j: K(source[i], dest[j])), (N, N), dtype=int)

N = 512
A = interaction_matrix(N)
relative_singular_tolerance = 1e-10
MB = MatrixButterfly(relative_singular_tolerance, decomposition='svd')
butterfly = multidimensional_butterfly(MB, A, 4, [(0, 1), (1, 0)])

rel_err = np.linalg.norm(A - MB.contract(butterfly)) / np.linalg.norm(A)
print(rel_err)
fig, axs = plt.subplots(1, len(butterfly[0]))
for L, ax in zip(butterfly[0], axs):
    ax.spy(L)
fig.suptitle(f"Butterfly factorization (relative error {rel_err})")
fig.show()

