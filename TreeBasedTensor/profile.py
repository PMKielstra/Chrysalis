from math import floor

BOTH = 0
UP = -1
DOWN = 1

class Profile:
    """Stores all the parameters necessary for a factorization."""

    def __init__(self, N, dimens, eps, levels, distance=1, direction=BOTH, subsamples = 20, translation_invariant=False, as_matrix = False, use_fake = False, verbose = False, boost_subsamples = True, processes=None):
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
        self.dsquared = distance ** 2
        self.direction = direction
        
        if boost_subsamples and not translation_invariant:
            self.subsamples = max(1, floor(subsamples ** (2 / self.dimens)))
        else:
            self.subsamples = subsamples
        if translation_invariant:
            assert direction == BOTH
        self.translation_invariant = translation_invariant
        
        self.as_matrix = as_matrix
        self.use_fake = use_fake
        self.verbose = verbose
        self.processes = processes
        self.axis_roll = 0
        self.off_split_number = 2 ** levels if direction == BOTH else 1
        
    def factor_index(self, is_source):
        return 0 if is_source else self.dimens

    def set_axis_roll(self, axis_roll):
        self.axis_roll = axis_roll
