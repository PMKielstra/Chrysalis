from ButterflyInterface import ButterflyInterface

class NoneButterfly(ButterflyInterface):
    """A ButterflyInterface class that returns None, or variants thereof.  Used for testing that the abstract butterfly doesn't break any abstraction barriers."""

    def shape(self, A, axis):
        return 100

    def split(self, A, axis, n):
        return [None]

    def factor(self, A, singular_values_left):
        return None, None

    def merge(self, As, axis=0):
        return None

    def diag(self, us, dimens=1):
        return None

    def compose(self, L, A, compose_left):
        return None

    def join(self, L1, L2):
        return None

    def transpose(self, A, ax0, ax1):
        return None

    def contract(self, As):
        return None
