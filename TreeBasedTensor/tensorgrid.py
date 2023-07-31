import numpy as np
from utils import tensorprod, multilevel_access, czip

class TensorGrid:
    def __recursive_split(self, A, splits, dimen):
        split_A = np.array_split(A, splits, axis=dimen)
        if dimen == A.ndim - 1:
            return split_A
        return [self.__recursive_split(B, splits, dimen + 1) for B in split_A]
    
    def __init__(self, A, splits, split_A=None, dimens=None):
        if split_A is not None and dimens is not None:
            self.split_A = split_A
            self.dimens = dimens
        else:
            self.split_A = self.__recursive_split(A, splits, 0)
            self.dimens = A.ndim

    def __recursive_get(self, As, position):
        if len(position) == 0:
            return As
        return self.__recursive_get(As[position[0]], position[1:])

    def get(self, position):
        return self.__recursive_get(self.split_A, position)

    def __recursive_add(self, As, Bs, level):
        if level == self.dimens:
            return As + Bs
        return [self.__recursive_add(A, B, level + 1) for A, B in czip(As, Bs)]

    def __add__(self, other):
        assert self.dimens == other.dimens
        return TensorGrid(None, None, self.__recursive_add(self.split_A, other.split_A, 0), self.dimens)

    def __recursive_multiply(self, get_U_at, A_grid, axis, position, verbose):
        if len(position) == self.dimens:
            return tensorprod(get_U_at(position), A_grid, axis, False) # TODO: Bring back verbose
        return [self.__recursive_multiply(get_U_at, A, axis, position + [i], verbose) for i, A in enumerate(A_grid)]

    def axis_multiply(self, get_U_at, axis, verbose=False):
        self.split_A = self.__recursive_multiply(get_U_at, self.split_A, axis, [], verbose)

    def __merge_block_at(self, block_position, elt_position):
        if len(block_position) == 0:
            return multilevel_access(self.split_A, elt_position)
        return [self.__merge_block_at(block_position[1:], elt_position + [2 * block_position[0]]), self.__merge_block_at(block_position[1:], elt_position + [2 * block_position[0] + 1])]

    def __recursive_merge(self, level, position):
        if level == self.dimens:
            return np.block(self.__merge_block_at(position, []))
        dimension_length = len(multilevel_access(self.split_A, [0] * (level)))
        return [self.__recursive_merge(level + 1, position + [i]) for i in range(dimension_length // 2)]

    def merge_once(self):
        self.split_A = self.__recursive_merge(0, [])

    def __recursive_to_dict(self, As, position):
        if len(position) == self.dimens:
            return {tuple(position): As}
        positions_dict = {}
        for position_result in (self.__recursive_to_dict(A, position + [i]) for i, A in enumerate(As)):
            positions_dict.update(position_result)
        return positions_dict

    def get_positions_dict(self):
        return self.__recursive_to_dict(self.split_A, [])

    def __recursive_from_dict(self, As_dict, side_length, dimens, position):
        if len(position) == dimens:
            return As_dict[tuple(position)]
        return [self.__recursive_from_dict(As_dict, side_length, dimens, position + [i]) for i in range(side_length)]

    def fill_from_positions_dict(self, As_dict, side_length, dimens):
        self.dimens = dimens
        self.split_A = self.__recursive_from_dict(As_dict, side_length, dimens, [])
