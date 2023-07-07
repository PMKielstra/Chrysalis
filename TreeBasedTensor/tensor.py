import numpy as np

def K_from_coords(N, coords_list):
    """Get a subtensor of K."""
    coords = np.meshgrid(*coords_list, indexing='ij')
    halflen = len(coords_list) // 2
    leftstack = np.stack(coords[:halflen], axis=0)
    rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
    return np.exp(1j * N * np.pi * norm) / norm

def AK(A, N, coords_list):
    """Carry out tensor compression along the axes of A and the first axes of the subtensor of K given by coords_list."""
    return np.tensordot(A, K_from_coords(N, coords_list), axes=A.ndim)

def AK_true(A, N):
    """Wrapper for AK which uses the full coords list."""
    return AK(A, N, [list(range(N))] * (A.ndim * 2))
