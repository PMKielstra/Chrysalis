import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def K(r1, r2, k):
    """e^{k pi i |r1 - r2|} / |r1 - r2|"""
    return np.exp(1j * k * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def interaction_tensor_matrix(N, C, l):
    """C is the width of the lowest element of the quadtree; N is the width of the overall tensor."""
    source_width = C * (2 ** l)
    assert N % (2 ** l) == 0
    observer_width = N // (2 ** l)
    points_1d = np.linspace(0, 1, N)
    xgrid, ygrid = np.meshgrid(points_1d, points_1d)
    subsample = 20
    subsamplesj = np.random.default_rng().integers(0, N, subsample)
    subsamplesk = np.random.default_rng().integers(0, observer_width, subsample)
    subsamplesl = np.random.default_rng().integers(0, observer_width, subsample)
    def Kijkl(i, j, k, l):
        return K( \
                np.array((0, xgrid[i, subsamplesj[j]], ygrid[i, subsamplesj[j]])), \
                np.array((1, xgrid[subsamplesk[k], subsamplesl[l]], ygrid[subsamplesk[k], subsamplesl[l]])), \
                N # N ~ k
            )
    tensor = np.fromfunction(np.vectorize(Kijkl), (source_width, subsample, subsample, subsample), dtype=int)
    return np.reshape(tensor, (source_width, subsample * subsample * subsample))

def singular_values(A):
    return np.linalg.svd(A, compute_uv=False)

def plot_with_tols(X, S_values, tols):
    for tol in tols:
        plt.plot(X, [np.sum(S > S[0] * tol) for S in S_values], label=f"tol={tol}")

# Fix fairly arbitrary values
C = 16
N = 1024
levels = list(range(1, 5))
S_values = [singular_values(interaction_tensor_matrix(N, C, l)) for l in tqdm(levels)]
plot_with_tols(levels, S_values, [1e-2, 1e-6, 1e-10])
plt.xlabel("Level")
plt.ylabel("Estimated rank")
plt.legend()
plt.show()
