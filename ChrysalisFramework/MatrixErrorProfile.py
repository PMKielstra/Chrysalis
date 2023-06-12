from AbstractButterfly import multi_axis_butterfly
from MatrixButterfly import MatrixButterfly
import numpy as np
from matplotlib import pyplot as plt

from NoneButterfly import NoneButterfly

def make_K(n):
    def K(r1, r2):
        return np.exp(1j * n * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)
    return K

def interaction_matrix(N):
    source = np.array([[0, x] for x in np.linspace(0, 1, N)])
    dest = np.array([[1, x] for x in np.linspace(0, 1, N)])
    K = make_K(N)
    return np.fromfunction(np.vectorize(lambda i, j: K(source[i], dest[j])), (N, N), dtype=int)

Ns = [32, 64, 128, 256, 512, 1024, 2048]
relative_singular_tolerance = 1e-10
decomposition = 'svd'

MB = MatrixButterfly(relative_singular_tolerance, decomposition)
errors = []
for i, N in enumerate(Ns):
    print(f"Now profiling N={N} ({i+1} of {len(Ns)})") 
    A = interaction_matrix(N)
    butterfly = multi_axis_butterfly(MB, A, 4, [(0, 1), (1, 0)])
    errors.append(np.linalg.norm(A - MB.contract(butterfly)) / np.linalg.norm(A))

plt.plot(Ns, errors)
plt.xlabel("N")
plt.ylabel("Error")
plt.show()
