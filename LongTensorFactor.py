import numpy as np
import scipy as sp
import scipy.linalg.interpolative as interp
from tqdm import tqdm
from random import sample
from tensorly import unfold
from matplotlib import pyplot as plt
from functools import reduce, cache
from math import floor, prod
import itertools
import time

from SliceManagement import SliceTree, split_to_tree, Multirange, EXTEND, IGNORE

def evaluate(N, levels):
    eps = 1e-6

    def K(r1, r2):
        """Tensor kernel representing the interaction between the two points r1 and r2.  Currently unused, but kept around as a reference."""
        d = np.linalg.norm(r1 - r2)
        return np.exp(1j * N * np.pi * d) / d

    def K_from_coords(coords_list):
        coord_grid = np.array(np.fromfunction(np.vectorize(lambda i, j, k, l: (0 - 1, (coords_list[0][i] - coords_list[2][k]) / (N-1), (coords_list[1][j] - coords_list[3][l]) / (N-1))), (len(r) for r in coords_list), dtype=int))
        norm = np.linalg.norm(coord_grid, axis=0)
        return np.exp(1j * N * np.pi * norm) / norm

    n_subsamples = 20
    def np_sample(range_x):
        """Subsample a list at random."""
        return np.array(sample(list(range_x), min(n_subsamples, len(range_x))))

    def ss_row_id(multirange, factor_index):
        """Carries out a subsampled row ID for a tensor, unfolded along factor_index."""
        
        # Step 1: Subsample all rows but the factor index
        sample_apply = [np_sample] * len(multirange)
        sample_apply[factor_index] = lambda x: x
        sampled_ranges = multirange.apply(sample_apply)

        # Step 2: Set up a tensor from the points chosen by the subsampling
        A = K_from_coords(sampled_ranges)

        # Step 3: Unfold the tensor and carry out ID
        unfolded = unfold(A, factor_index).T # The transpose here is because we want row decompositions, not column, but Scipy only does column decompositions, not rows.
        k, idx, proj = interp.interp_decomp(unfolded, eps)
        R = interp.reconstruct_interp_matrix(idx, proj)

        # Step 4: Map the rows chosen by the ID, which are a subset of [1, ..., len(multirange[factor_index])], back to a subset of the relevant actual rows
        old_rows = multirange[factor_index]
        new_rows = old_rows[idx[:k]]
        return new_rows, R.T

    def compute_matrices(level, multirange, factor_index):
        """Carry out a multi-level, single-axis butterfly, returning a subset of rows along that axis along with a list of interpolation matrices."""
        if level == 0:
            new_rows, R = ss_row_id(multirange, factor_index)
            return [new_rows], [R]
        next_steps = multirange.next_steps()
        next_results = [compute_matrices(level - 1, step, factor_index) for step in next_steps]
        next_rows = np.concatenate([row[-1] for row, U in next_results])
        Us = [sp.linalg.block_diag(*UU) for UU in zip(*(U for row, U in next_results))]
        rows = [np.concatenate(RR) for RR in zip(*(row for row, U in next_results))]
        new_rows, new_U = ss_row_id(multirange.overwrite(SliceTree(next_rows), factor_index), factor_index)
        return rows + [new_rows], Us + [new_U]

    class FactorProfile:
        dimensions: int
        factor_indices: list
        position_indices: list
        position_len: int
        levels: int
        splits_per_level: int

        def __init__(self, dimensions=4, factor_indices=None, position_indices=None, levels=2, splits_per_level=2):
            self.dimensions, self.levels, self.splits_per_level = dimensions, levels, splits_per_level
            if factor_indices is None:
                factor_indices = list(range(dimensions // 2))
            if position_indices is None:
                position_indices = list(range(factor_indices[-1], dimensions))
            self.position_len = len(position_indices)
            assert len(factor_indices) + self.position_len <= dimensions
            self.factor_indices, self.position_indices = factor_indices, position_indices
            self.split_pattern = [splits_per_level if i in position_indices else IGNORE for i in range(dimensions)]
            self.deepest_split = splits_per_level ** levels

    class FactorTree:
        def __init__(self, rows, matrix, position):
            self.rows = rows
            self.matrix = matrix
            self.position = position
            self.multirange = None
            self.children = []

    def factor_to_tree(rows_list, multirange, factor_index, profile):
        new_rows, Rs = [], []
        for r in rows_list:
            row, R = ss_row_id(multirange.overwrite(SliceTree(r), factor_index), factor_index)
            new_rows.append(row)
            Rs.append(R)
        tree = FactorTree(np.concatenate(new_rows), sp.linalg.block_diag(*Rs), tuple(multirange[i].position for i in profile.position_indices))
        if len(rows_list) > 1:
            merged_rows = [np.concatenate((a, b)) for a, b in zip(new_rows[::2], new_rows[1::2])]
            next_steps = multirange.next_steps()
            tree.children = [factor_to_tree(merged_rows, step, factor_index, profile) for step in next_steps]
        else:
            tree.multirange = multirange
        return tree

    class MultiFactorTree:
        def __init__(self, triples, position, multirange):
            self.position = position
            self.triples = triples
            self.multirange = multirange
            self.children = []

    def merge_trees(trees, factor_indices):
        new_tree = MultiFactorTree([(index, tree.rows, tree.matrix) for index, tree in zip(factor_indices, trees)], trees[0].position, trees[0].multirange)
        new_tree.children = [
            merge_trees(t, factor_indices) for t in zip(*(tree.children for tree in trees))
        ]
        return new_tree

    def factor_full(profile):
        rows_list = np.array_split(list(range(N)), profile.deepest_split)
        section_ranges = Multirange([SliceTree(list(range(N))) for _ in range(profile.dimensions)], profile.split_pattern)
        trees = [factor_to_tree(rows_list, section_ranges, factor_index, profile) for factor_index in profile.factor_indices]
        return merge_trees(trees, profile.factor_indices)

    def matrix_to_chunks(A, tree):
        processed_matrix = A
        for index, _rows, matrix in tree.triples:
            processed_matrix = np.swapaxes(np.tensordot(matrix, processed_matrix, (0, index)), 0, index)
        if tree.children == []:
            return {tree.position: ([row for _index, row, _matrix in tree.triples], processed_matrix, tree.multirange)}
        return {k: v for child in tree.children for k, v in matrix_to_chunks(processed_matrix, child).items()}

    def chunks_times_tensor(chunks, profile):
        split = np.array_split(list(range(N)), profile.deepest_split)
        new_chunks = {}
        for position in itertools.product(range(profile.deepest_split), repeat=profile.position_len):
            rows_list, matrix, multirange = chunks[position]
            for index, rows in zip(profile.factor_indices, rows_list):
                multirange = multirange.overwrite(SliceTree(rows), index)
            tensor = K_from_coords(list(multirange))
            new_matrix = np.tensordot(matrix, tensor, axes=2)
            new_chunks[position] = new_matrix
        return new_chunks

    def reconstitute(chunks, profile):
        def to_list_of_lists(level, partial_position):
            if level == profile.position_len:
                return chunks[tuple(partial_position)]
            return [to_list_of_lists(level + 1, partial_position + [i]) for i in range(profile.deepest_split)]
        return np.block(to_list_of_lists(0, []))

    def size(tree, profile):
        matrix_size = sum(prod(matrix.shape) for _index, _row, matrix in tree.triples)
        if tree.children == []:
            tensor_size = prod(len(row) for _index, row, _matrix in tree.triples) \
                * ((N // profile.deepest_split) ** profile.position_len) \
                * (N ** (profile.dimensions - profile.position_len - len(tree.triples)))
            return tensor_size, tensor_size + matrix_size
        children = (size(child, profile) for child in tree.children)
        tensor_size_only, tensor_size_with_matrix_size = 0, 0
        for tso, tsms in children:
            tensor_size_only += tso
            tensor_size_with_matrix_size += tsms
        return tensor_size_only, tensor_size_with_matrix_size + matrix_size

    def max_rank(tree):
        if tree.children == []:
            return [tree.triples[0][2].shape[1]]
        rank_children = [max_rank(child) for child in tree.children]
        return [tree.triples[0][2].shape[1]] + [max(r[i] for r in rank_children) for i in range(len(rank_children[0]))]

    profile = FactorProfile(factor_indices = [0, 1], position_indices = [2, 3], levels = levels)

    tree = factor_full(profile)
    A = np.random.rand(N, N)
    chunks = matrix_to_chunks(A, tree)
    tensored_chunks = chunks_times_tensor(chunks, profile)
    compressed_result = reconstitute(tensored_chunks, profile)
##    return max_rank(tree)
##    return compressed_result.shape
    print("Now building full tensor...")
    full_tensor = K_from_coords(tuple(tuple(range(N)) for _ in range(4)))
    full_result = np.tensordot(A, full_tensor, axes=2)
    return np.linalg.norm(full_result - compressed_result) / np.linalg.norm(full_result)
    

print(evaluate(41, 3))
