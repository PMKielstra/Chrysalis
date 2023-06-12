from AbstractButterfly import single_axis_butterfly, multi_axis_butterfly
from MatrixButterfly import MatrixButterfly
import numpy as np
from matplotlib import pyplot as plt

def K(r1, r2):
    return np.exp(1j * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def interaction_matrix(N):
    source = np.array([[0, x] for x in np.linspace(0, 1, N)])
    dest = np.array([[1, x] for x in np.linspace(0, 1, N)])
    return np.fromfunction(np.vectorize(lambda i, j: K(source[i], dest[j])), (N, N), dtype=int)

N = 256
A = interaction_matrix(N)
relative_singular_tolerance = 1e-6
MB = MatrixButterfly(relative_singular_tolerance, decomposition='svd')
butterfly = single_axis_butterfly(MB, A, 64, 1, 0)

rel_err = np.linalg.norm(A - MB.contract(butterfly)) / np.linalg.norm(A)
for i in butterfly:
    fig, axs = plt.subplots(1, len(butterfly[i]))
    for L, ax in zip(reversed(butterfly[i]), axs):
        ax.spy(L)
    fig.suptitle(f"Butterfly factorization (relative error {rel_err})")
    fig.show()

