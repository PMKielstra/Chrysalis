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
    def factor(self, A, singular_values_left):
        """Truncated SVD factorization for A.  Return (U.S, V^T) if singular_values_left, otherwise (U, S.V^T)."""
        pass

    @abstractmethod
    def diag(self, us, dimens=1):
        """The diagonal matrix [[us[0], 0, 0, ...], [0, us[1], 0, ...], [0, 0, us[2], ...], ...].  The us here are given as a list."""
        pass

    @abstractmethod
    def merge(self, As, axis=0):
        """Stack the As along the given axis.  The As here are given as a list."""
        pass

    @abstractmethod
    def compose(self, L, A, compose_left):
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

    def multiply(self, A, B):
        """Freebie: multiply base matrices without using factorizations."""
        empty = self.compose(None, None, False)
        return self.apply(self.compose(empty, A, False), B)

    def recursive_merge(self, us, axes):
        """Freebie: merge recursively in multiple dimensions."""
        if len(axes) == 1:
            return self.merge(us, axes[0])
        else:
            return self.merge([self.recursive_merge(u, axes[1:]) for u in us], axes[0])
