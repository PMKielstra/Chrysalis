import numpy as np
import random

def czip(*ls):
    """Zip, but assert that all lists involved have the same length."""
    for l in ls:
        assert len(l) == len(ls[0])
    return zip(*ls)

n_subsamples = 20
def subsample(range_x):
    """Subsample a list at random."""
    if len(range_x) <= n_subsamples:
        return range_x
    return np.array(random.sample(list(range_x), min(n_subsamples, len(range_x))))

def slice_by_index(A, down_index, i=0):
    if i == len(down_index):
        return A
    slice_list = [slice(None)] * i + [down_index[i]]
    return slice_by_down_index(A[tuple(slice_list)], down_index, i + 1)

def multilevel_access(l, indices, assert_single_element=False):
    if len(indices) == 0:
        return l
    if assert_single_element:
        assert len(l) == 1
    return multilevel_access(l[indices[0]], indices[1:])

def multilevel_enumerate(l, levels):
    if levels == 0:
        return [([], l)]
    if levels == 1:
        return (([i], ll) for i, ll in enumerate(l))
    return (([j] + pos, elt) for j, ll in enumerate(l) for pos, elt in multilevel_enumerate(ll, levels - 1))

def multilevel_flatten(l):
    if not isinstance(l, list):
        return [l]
    if not isinstance(l[0], list):
        return l
    return [c for ll in l for c in multilevel_flatten(ll)]
