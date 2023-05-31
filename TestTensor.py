from AbstractButterfly import single_axis_butterfly, multi_axis_butterfly
from MatrixButterfly import MatrixButterfly
import numpy as np
from matplotlib import pyplot as plt

def K(r1, r2):
    return np.exp(1j * np.pi * np.linalg.norm(r1 - r2)) / np.linalg.norm(r1 - r2)

def interaction_tensor(N):
    xs = np.linspace(0, 1, N)
    xgrid, ygrid = np.meshgrid(xs, xs)
    return np.fromfunction(np.vectorize(lambda i, j, k, l: K(np.array((0, xgrid[i, j], ygrid[i, j])), np.array((1, xgrid[k, l], ygrid[k, l])))), (N, N, N, N), dtype=int)


N = 256
A = interaction_tensor(N)
relative_singular_tolerance = 1e-10
