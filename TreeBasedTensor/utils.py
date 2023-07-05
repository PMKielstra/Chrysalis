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
