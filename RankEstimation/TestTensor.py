import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def K(r1, r2, k):
    return np.exp(1j * k * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def interaction_tensor(N, C, subsample):
    xs = np.linspace(0, 1, N)
    xgrid, ygrid = np.meshgrid(xs, xs)
    subsamplesj = np.random.default_rng().integers(0, N, subsample)
    subsamplesk = np.random.default_rng().integers(0, N, subsample)
    subsamplesl = np.random.default_rng().integers(0, N, subsample)
    return np.fromfunction(np.vectorize(lambda i, j, k, l: K(np.array((0, xgrid[i, subsamplesj[j]], ygrid[i, subsamplesj[j]])), np.array((1, xgrid[subsamplesk[k], subsamplesl[l]], ygrid[subsamplesk[k], subsamplesl[l]])), N)), (C, subsample, subsample, subsample), dtype=int)

def sqrt_interaction_tensor(SqN, subsample):
    N = SqN * SqN
    xs = np.linspace(0, 1, N)
    xgrid, ygrid = np.meshgrid(xs, xs)
    subsamples = np.random.default_rng().integers(0, N, subsample)
    return np.fromfunction(np.vectorize(lambda i, j, k, l: K(np.array((0, xgrid[i, j], ygrid[i, j])), np.array((1, xgrid[k, subsamples[l]], ygrid[k, subsamples[l]])), N)), (SqN, SqN, SqN, subsample), dtype=int)

def check_low_rank(N, C, subsample):
    A = interaction_tensor(N, C, subsample)
    B = np.reshape(A, (C, subsample * subsample * subsample))
    S = np.linalg.svd(B, compute_uv=False)
    return S

def sqrt_low_rank(SqN, subsample):
    A = sqrt_interaction_tensor(SqN, subsample)
    B = np.reshape(A, (SqN * SqN * SqN, subsample))
    S = np.linalg.svd(B, compute_uv=False)
    return S

tolerances = [1e-2, 1e-4, 1e-6, 1e-8]
SqNs = [2, 4, 8, 16, 32, 64]
S_values = []
for SqN in tqdm(SqNs):
    S_values.append(sqrt_low_rank(SqN, 40))


for tol in tolerances:
    plt.plot(SqNs, [np.sum(S > S[0] * tol) for S in S_values], label=f"tol={tol}")
plt.xlabel("sqrt(N)")
plt.ylabel("Numerical rank as estimated by random approximation")
plt.title(f"Numerical rank of a three-way sqrt(N) slice through an N^4 tensor")
plt.legend()
plt.show()
