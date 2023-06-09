from math import floor

BOTH = 0
UP = -1
DOWN = 1

class Profile:
    """Stores all the parameters necessary for a factorization."""

    def __init__(self, N, dimens, eps, levels, direction=BOTH, subsamples = 20, as_matrix = False, verbose = False, boost_subsamples = True, processes=None):
        assert N > 0
        assert dimens > 0
        assert eps < 1
        assert direction in (BOTH, UP, DOWN)

        self.N = N ** dimens if as_matrix else N
        self.true_N = N
        self.dimens = 1 if as_matrix else dimens
        self.true_dimens = dimens
        self.eps = eps
        self.levels = levels * (dimens if as_matrix else 1)
        self.direction = direction
        if boost_subsamples:
            self.subsamples = max(1, floor(subsamples ** (2 / dimens)))
        else:
            self.subsamples = subsamples
        self.as_matrix = as_matrix
        self.verbose = verbose
        self.processes = processes
        self.factor_source = 0 # TODO: add the possibility to factor along different axes.
        self.factor_observer = self.dimens
        self.off_split_number = 2 ** levels if direction == BOTH else 1
        
    def factor_index(self, is_source):
        return self.factor_source if is_source else self.factor_observer
