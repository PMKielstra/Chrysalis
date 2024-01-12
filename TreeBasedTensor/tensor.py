from math import floor, log2

import numpy as np

def sum_pow(t):
    return t + t ** 2 + t ** 3

def mod_stack(profile, coords):
    stack = [np.mod(coords, 2)]
    logN = floor(log2(profile.N))
    for _ in range(logN - 1):
        coords = (coords - stack[-1]) // 2
        stack.append(np.mod(coords, 2))
    coords = (coords - stack[-1]) // 2
    assert max(np.ravel(coords)) == 0
    assert min(np.ravel(coords)) == 0 
    split_stacks = [stack[i::profile.true_dimens] for i in range(profile.true_dimens)]
    split_pos = [sum((2 ** i) * x for i, x in enumerate(s)) for s in split_stacks]
    return np.float64(np.stack(split_pos, axis=0))

def K_from_coords_green(profile, leftstack, rightstack):
    """Get a subtensor of K."""
    if profile.flat:
        rightstack += (profile.true_N - 1) * (1 + profile.distance)
    norm = np.sqrt((0 if profile.flat else profile.dsquared) + np.sum(((leftstack - rightstack) / (profile.true_N - 1)) ** 2, axis=0))
    norm = np.swapaxes(norm, 0, profile.axis_roll)
    norm = np.swapaxes(norm, profile.dimens, profile.dimens + profile.axis_roll)

    if profile.kill_trans_inv:
        kti = 1 - 0.00001 * np.cos(np.sum(leftstack, axis=0))
        kti = np.swapaxes(kti, 0, profile.axis_roll)
        kti = np.swapaxes(kti, profile.dimens, profile.dimens + profile.axis_roll)
    else:
        kti = 1
    
    return np.exp(-1j * profile.true_N * np.pi * norm * kti) / norm

def binary_inversion(profile, x):
    y = np.zeros_like(x)
    n = floor(log2(profile.N)) - 1
    while np.max(x) > 0:
        assert n >= 0
        y += (2 ** n) * (x % 2)
        x = (x - (x % 2)) // 2
        n -= 1
    return y

def K_from_coords_fourier(profile, leftstack, rightstack):
    leftstack, rightstack = binary_inversion(profile, leftstack), binary_inversion(profile, rightstack)
    dot_prod = np.sum(leftstack * rightstack, axis=0)
    dot_prod = np.swapaxes(dot_prod, 0, profile.axis_roll)
    dot_prod = np.swapaxes(dot_prod, profile.dimens, profile.dimens + profile.axis_roll)

    return np.exp(-2j * np.pi * dot_prod / (profile.true_N))

def K_from_coords_random_fourier(profile, leftstack, rightstack):
    leftstack, rightstack = binary_inversion(profile, leftstack), binary_inversion(profile, rightstack)
    leftstack, rightstack = profile.rand_left[leftstack.astype(int)], profile.rand_right[rightstack.astype(int)]
    
    dot_prod = np.sum(leftstack * rightstack, axis=0)
    dot_prod = np.swapaxes(dot_prod, 0, profile.axis_roll)
    dot_prod = np.swapaxes(dot_prod, profile.dimens, profile.dimens + profile.axis_roll)

    return np.exp(-2j * np.pi * dot_prod)
    
def K_from_coords(profile, coords_list):
    coords_list[0], coords_list[profile.axis_roll] = coords_list[profile.axis_roll], coords_list[0]
    coords_list[profile.dimens], coords_list[profile.dimens + profile.axis_roll] = coords_list[profile.dimens + profile.axis_roll], coords_list[profile.dimens]
    if profile.as_matrix:
        assert len(coords_list) == 2
        x, y = np.meshgrid(range(len(coords_list[0])), range(len(coords_list[1])), indexing='ij')
        leftstack, rightstack = mod_stack(profile, np.array(coords_list[0])[x]), mod_stack(profile, np.array(coords_list[1])[y])
    else:
        coords = np.meshgrid(*coords_list, indexing='ij')
        halflen = len(coords_list) // 2
        leftstack = np.float64(np.stack(coords[:halflen], axis=0))
        rightstack = np.float64(np.stack(coords[halflen:], axis=0))

    if profile.fourier:
        if profile.nonuniform:
            return K_from_coords_random_fourier(profile, leftstack, rightstack)
        return K_from_coords_fourier(profile, leftstack, rightstack)
    else:
        return K_from_coords_green(profile, leftstack, rightstack)

def AK(profile, A, coords_list):
    """Carry out tensor compression along the axes of A and the first axes of the subtensor of K given by coords_list."""
    return np.tensordot(A, K_from_coords(profile, coords_list), axes=A.ndim)

def AK_true(profile, A):
    """Wrapper for AK which uses the full coords list."""
    return AK(profile, A, [list(range(profile.N))] * (A.ndim * 2))
