import numpy as np

def mod_stack(profile, coords):
    stack = [np.mod(coords, 2)]
    for _ in range(profile.true_dimens * (profile.levels + 3)):
        coords = (coords - stack[-1]) // 2
        stack.append(np.mod(coords, 2))
    assert max(np.ravel(coords)) == 0
    split_stacks = [stack[i::profile.true_dimens] for i in range(profile.true_dimens)]
    split_pos = [sum((2 ** i) * x for i, x in enumerate(s)) for s in split_stacks]
    return np.stack(split_pos, axis=0)

def K_from_coords(profile, coords_list):
    """Get a subtensor of K."""
    coords_list[0], coords_list[profile.axis_roll] = coords_list[profile.axis_roll], coords_list[0]
    coords_list[profile.dimens], coords_list[profile.dimens + profile.axis_roll] = coords_list[profile.dimens + profile.axis_roll], coords_list[profile.dimens]
    if profile.as_matrix:
        assert len(coords_list) == 2
        x, y = np.meshgrid(range(len(coords_list[0])), range(len(coords_list[1])), indexing='ij')
        leftstack, rightstack = mod_stack(profile, np.array(coords_list[0])[x]), mod_stack(profile, np.array(coords_list[1])[y])
    else:
        coords = np.meshgrid(*coords_list, indexing='ij')
        halflen = len(coords_list) // 2
        leftstack = np.stack(coords[:halflen], axis=0)
        rightstack = np.stack(coords[halflen:], axis=0)
    if profile.flat:
        rightstack += (profile.true_N - 1) * (1 + profile.distance)
    norm = np.sqrt((0 if profile.flat else profile.dsquared) + np.sum(((leftstack - rightstack) / (profile.true_N - 1)) ** 2, axis=0))
    norm = np.swapaxes(norm, 0, profile.axis_roll)
    norm = np.swapaxes(norm, profile.dimens, profile.dimens + profile.axis_roll)
    return np.exp(-1j * profile.true_N * np.pi * norm) / norm

def AK(profile, A, coords_list):
    """Carry out tensor compression along the axes of A and the first axes of the subtensor of K given by coords_list."""
    return np.tensordot(A, K_from_coords(profile, coords_list), axes=A.ndim)

def AK_true(profile, A):
    """Wrapper for AK which uses the full coords list."""
    return AK(profile, A, [list(range(profile.N))] * (A.ndim * 2))
