from abc import ABC, abstractmethod

class ButterflyInterface(ABC):
    """The most basic operation set required to carry out a butterfly factorization.  All docstrings are written as if we were factorizing matrices, but this should in theory work for matrices, tensors, bananas, whatever."""
    
    @abstractmethod
    def shape(self, A, axis):
        """The size of the matrix A along the given axis."""
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
    def compose(self, L, A, axis):
        """[A] + L, L + [A], or [] if L is None."""
        pass

    @abstractmethod
    def join(self, L1, L2):
        """Blindly join factorizations"""
        pass

    @abstractmethod
    def transpose(self, A, ax0, ax1):
        """Swap ax0 and ax1 in A."""
        pass

    @abstractmethod
    def contract(self, As):
        """Contract the As (the result of calling compose) to a single matrix."""
        pass

    def apply(self, As, X):
        """Freebie: apply the As to a particular X."""
        return self.contract(self.compose(As, X, False))

    def multiply(self, A, B, axis=0):
        """Freebie: multiply base matrices without using factorizations."""
        empty = self.compose(None, None, False)
        return self.apply(self.compose(empty, A, axis), B)

    def recursive_stack(self, us, axes):
        """Freebie: stack recursively in multiple dimensions."""
        if len(axes) == 1:
            return self.stack(us, axes[0])
        else:
            return self.stack([self.recursive_stack(u, axes[1:]) for u in us], axes[0])
