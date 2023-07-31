import numpy as np
import random

def tensorprod(A, K, axis, verbose=False):
    if verbose:
        print(A.shape, K.shape)
    K = np.tensordot(A, K, axes=(1, axis))
    for i in range(axis):
        K = np.swapaxes(K, i, i + 1)
    return K

def without(l, i):
    """All of l except for l[i]"""
    return [ll for j, ll in enumerate(l) if j != i]

def czip(*ls):
    """Zip, but assert that all lists involved have the same length."""
    for l in ls:
        assert len(l) == len(ls[0])
    return zip(*ls)

def subsample(range_x, n_subsamples):
    """Subsample a list at random."""
    if len(range_x) <= n_subsamples:
        return range_x
    return np.array(random.sample(list(range_x), min(n_subsamples, len(range_x))))

def slice_by_index(A, down_index, i=0):
    """Compute A[down_index[0]][:, down_index[1]][:, :, down_index[2]]... (which is different to A[down_index[0], down_index[1], ...])."""
    if i == len(down_index):
        return A
    slice_list = [slice(None)] * i + [down_index[i]]
    return slice_by_index(A[tuple(slice_list)], down_index, i + 1)

def multilevel_access(l, indices, assert_single_element=False):
    """Find l[indices[0]][indices[1]][indices[2]]..., optionally asserting along the way that there is only one element at each level."""
    if len(indices) == 0:
        return l
    if assert_single_element:
        assert len(l) == 1
    return multilevel_access(l[indices[0]], indices[1:])

def multilevel_enumerate(l, levels):
    """Flatten a nested list to the given number of levels and enumerate it in a multi-levelled fashion.  For example, the list [[A, B], [C]] would become [([0, 0], A), ([0, 1], B), ([1, 0], C)] when enumerated with levels=2."""
    if levels == 0:
        return [([], l)]
    if levels == 1:
        return (([i], ll) for i, ll in enumerate(l))
    return (([j] + pos, elt) for j, ll in enumerate(l) for pos, elt in multilevel_enumerate(ll, levels - 1))

def multilevel_flatten(l):
    """Totally flatten a list.  If passed anything other than a list, return that item wrapped in a one-element list, so you can always assume that the output from this function will be a list of depth 1."""
    if not isinstance(l, list):
        return [l]
    if not isinstance(l[0], list):
        return l
    return [c for ll in l for c in multilevel_flatten(ll)]
