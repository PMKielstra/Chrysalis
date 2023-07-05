import numpy as np

def K_from_coords(N, coords_list):
    coords = np.meshgrid(*coords_list, indexing='ij')
    halflen = len(coords_list) // 2
    leftstack = np.stack(coords[:halflen], axis=0)
    rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
    return np.exp(1j * N * np.pi * norm) / norm

def AK(A, N, coords_list):
    return np.tensordot(A, K_from_coords(N, coords_list), axes=2)

def AK_true(A, N):
    return AK(A, N, [list(range(N))] * 4)
