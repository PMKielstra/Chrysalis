import numpy as np

class AugmentedList:
    def store_list(self, aug_list):
        self.aug_list = aug_list

    def list(self):
        return self.aug_list

    def __getitem__(self, subscript):
        return self.aug_list[subscript]

    def __repr__(self):
        return self.aug_list.__repr__()

    def __len__(self):
        return len(self.aug_list)

    def __iter__(self):
        return self.aug_list.__iter__()

class SliceTree(AugmentedList):
    def __init__(self, cols_list, parent=None):
        self.store_list(cols_list)
        self.parent = parent
    
    def split(self, n):
        splits = np.array_split(self.list(), n)
        return [SliceTree(new_list, self) for new_list in splits]

    def extend(self):
        return self.parent

def split_to_tree(cols_list, levels, splits_per_level):
    trees = [SliceTree(cols_list)]
    for _ in range(levels):
        trees = [node for tree in trees for node in tree.split(splits_per_level)]
    return trees

EXTEND = -1
IGNORE = 0
class Multirange(AugmentedList):
    def __init__(self, ranges, split_pattern):
        assert(len(ranges) == len(split_pattern))
        self.store_list(ranges)
        self.split_pattern = split_pattern

    def __recursive_split(self, ranges, split_pattern):
        if len(ranges) == 0:
            return [[]]
        remainder = self.__recursive_split(ranges[1:], split_pattern[1:])
        if split_pattern[0] == EXTEND:
            return [[ranges[0].extend()] + i for i in remainder]
        elif split_pattern[0] == IGNORE:
            return [[ranges[0]] + i for i in remainder]
        else:
            split_range = ranges[0].split(split_pattern[0])
            return [[s] + i for s in split_range for i in remainder]

    def next_steps(self):
        return [Multirange(r, self.split_pattern) for r in self.__recursive_split(self.list(), self.split_pattern)]

    def apply(self, f):
        if callable(f):
            return [f(r) for r in self.list()]
        return [ff(r) for ff, r in zip(f, self.list())]

    def overwrite(self, new_val, index):
        return Multirange([(new_val if i == index else r) for i, r in enumerate(self.list())], self.split_pattern)
