import numpy as np

def mod_stack(profile, coords):
    stack = [np.mod(coords, profile.true_N)]
    for _ in range(profile.true_dimens - 1):
        coords = (coords - stack[-1]) / profile.true_N
        stack.append(np.mod(coords, profile.true_N))
    return np.stack(stack, axis=0)

def K_from_coords(profile, coords_list):
    """Get a subtensor of K."""
    if profile.as_matrix:
        assert len(coords_list) == 2
        x, y = np.meshgrid(range(len(coords_list[0])), range(len(coords_list[1])), indexing='ij')
        leftstack, rightstack = mod_stack(profile, np.array(coords_list[0])[x]), mod_stack(profile, np.array(coords_list[1])[y])
    else:
        coords = np.meshgrid(*coords_list, indexing='ij')
        halflen = len(coords_list) // 2
        leftstack = np.stack(coords[:halflen], axis=0)
        rightstack = np.stack(coords[halflen:], axis=0)
    norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (profile.true_N - 1)) ** 2, axis=0))
    return np.exp(1j * profile.true_N * np.pi * norm) / norm

def AK(profile, A, coords_list):
    """Carry out tensor compression along the axes of A and the first axes of the subtensor of K given by coords_list."""
    return np.tensordot(A, K_from_coords(profile, coords_list), axes=A.ndim)

def AK_true(profile, A):
    """Wrapper for AK which uses the full coords list."""
    return AK(A, N, [list(range(N))] * (A.ndim * 2))
