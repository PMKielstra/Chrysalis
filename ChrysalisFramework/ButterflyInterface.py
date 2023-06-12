from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce

class ButterflyInterface(ABC):
    """The most basic operation set required to carry out a butterfly factorization.  All docstrings are written as if we were factorizing matrices, but this should in theory work for matrices, tensors, bananas, whatever.  Includes built-in but overridable functionality for a "axis-dict factorization"."""
    
    @abstractmethod
    def shape(self, A, axis):
        """The size of the matrix A along the given axis."""
        pass

    @abstractmethod
    def multiply(self, A, B, axis=0):
        """Multiply A and B along the given axis."""
        pass
    
    @abstractmethod
    def split(self, A, axis, n):
        """Split A into n parts along the given axis."""
        pass

    @abstractmethod
    def factor(self, A, factor_axis, aux_axis):
        """Truncated factorization for A.  Exchanging factor_axis and aux_axis should swap the roles of the two resultant matrices (for instance, going from a QR to an RQ factorization)."""
        pass

    @abstractmethod
    def build_center(self, K, U, axis):
        """Combine a U matrix with a slice K of the original A matrix in order to build a center matrix.  How this is done is highly dependent on the chosen factorization."""
        pass

    @abstractmethod
    def diag(self, us, dimens=1):
        """The diagonal matrix [[us[0], 0, 0, ...], [0, us[1], 0, ...], [0, 0, us[2], ...], ...].  The us here are given as a list."""
        pass

    @abstractmethod
    def stack(self, As, axis=0):
        """Stack the As along the given axis.  The As here are given as a list."""
        pass

    @abstractmethod
    def transpose(self, A, ax0, ax1):
        """Swap ax0 and ax1 in A."""
        pass

    def compose(self, L, A, axis):
        """Add A to L along the given axis."""
        if L == None:
            return {}
        L = deepcopy(L)
        if axis not in L:
            L[axis] = []
        L[axis] = L[axis] + [A]
        return L

    def join(self, L1, L2):
        """Join factorizations at their ends."""
        axes = set().union(L1, L2)
        result = {}
        for axis in axes:
            result[axis] = L1.get(axis, []) + L2.get(axis, [])
        return result

    def contract(self, L):
        """Contract L (the result of calling compose) to a single matrix."""
        def reduce_along(axis):
            return reduce(lambda A, B: self.multiply(A, B, axis), (L[axis]))
        axes = list(L.keys())
        return reduce(lambda accum, i: self.multiply(accum, reduce_along(axes[i]), axes[i - 1]), range(1, len(axes)), reduce_along(axes[0]))

    def apply(self, As, X, axis):
        """Apply the As to a particular X along a particular axis."""
        return self.contract(self.compose(As, X, axis))

    def recursive_stack(self, us, axes):
        """Stack recursively in multiple dimensions."""
        if len(axes) == 1:
            return self.stack(us, axes[0])
        else:
            return self.stack([self.recursive_stack(u, axes[1:]) for u in us], axes[0])
