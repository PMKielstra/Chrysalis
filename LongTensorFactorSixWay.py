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
from SparseDiag import SparseDiag

t = 0
def tick():
    global t
    t = time.time()

def tock():
    global t
    return time.time() - t

def evaluate(N, levels):
    eps = 1e-6

    def K_from_coords(coords_list):
        coords = np.meshgrid(*coords_list, indexing='ij')
        halflen = len(coords_list) // 2
        leftstack = np.stack(coords[:halflen], axis=0)
        rightstack = np.stack(coords[halflen:], axis=0)
        norm = np.sqrt(1 + np.sum(((leftstack - rightstack) / (N - 1)) ** 2, axis=0))
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

    class FactorProfile:
        """A class to store basic information about a factorization.  Carries out various useful calculations when it's created.""" 
        dimensions: int
        factor_indices: list
        position_indices: list
        position_len: int
        levels: int
        splits_per_level: int
        split_pattern: list
        deepest_split: list

        def __init__(self, dimensions, factor_indices=None, position_indices=None, levels=2, splits_per_level=2):
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

    class MultiFactorTree:
        """Stores a point on the factorization tree.  Each triple in the list of triples is of the form (factor_index, rows_list, matrix)."""
        def __init__(self, triples, multirange):
            self.triples = triples
            self.multirange = multirange
            self.children = []
            self.subtensor = None

    def factor_to_tree(rows_lists, multirange, profile):
        """Given a list of lists of rows and a multirange to which those rows are associated, recursively factor the tensor and output a MultiFactorTree."""
        triples = []
        merged_rows = []
        for factor_index, rows_list in zip(profile.factor_indices, rows_lists):
            new_rows, Rs = [], []
            for r in rows_list:
                row, R = ss_row_id(multirange.overwrite(SliceTree(r), factor_index), factor_index)
                new_rows.append(row)
                Rs.append(R)
            triples.append((factor_index, np.concatenate(new_rows), SparseDiag(Rs)))
            if len(rows_lists[0]) > 1:
                merged_rows.append([np.concatenate((a, b)) for a, b in zip(new_rows[::2], new_rows[1::2])])
        tree = MultiFactorTree(triples, multirange)
        if len(rows_lists[0]) > 1:
            next_steps = multirange.next_steps()
            tree.children = [factor_to_tree(merged_rows, step, profile) for step in next_steps]
        else:
            for index, rows, _matrix in tree.triples:
                multirange = multirange.overwrite(SliceTree(rows), index)
            tree.subtensor = K_from_coords(list(multirange))
        return tree

    def factor_full(profile):
        """A quick entry point to the factor_to_tree function, which provides it with the initial arguments for the recursion."""
        section_ranges = Multirange([SliceTree(list(range(N))) for _ in range(profile.dimensions)], profile.split_pattern)
        return factor_to_tree(tuple(np.array_split(list(range(N)), profile.deepest_split) for _ in range(len(profile.factor_indices))), section_ranges, profile)

    def matrix_to_chunks(A, tree, profile):
        """Apply a factor tree to a matrix, producing a dict of chunks ready to be multiplied by their relevant sub-tensors."""
        processed_matrix = A
        for index, _rows, matrix in tree.triples:
            processed_matrix = np.moveaxis(np.tensordot(matrix, processed_matrix, (0, index)), range(index + 1), itertools.chain([index], range(index)))
            del matrix
        if tree.children == []:
            position = tuple(tree.multirange[i].position for i in profile.position_indices)
            return {position: (processed_matrix, tree.subtensor)}
        return {k: v for child in tree.children for k, v in matrix_to_chunks(processed_matrix, child, profile).items()}

    def chunks_times_tensor(chunks, profile):
        """Multiply the result of matrix_to_chunks by the important sub-tensors.  Return a dict of matrices (which is in a different format to the dict of chunks from before)."""
        split = np.array_split(list(range(N)), profile.deepest_split)
        new_chunks = {}
        for position in itertools.product(range(profile.deepest_split), repeat=profile.position_len):
            matrix, subtensor = chunks[position]
            new_matrix = np.tensordot(matrix, subtensor, axes=matrix.ndim)
            new_chunks[position] = new_matrix
        return new_chunks

    def reconstitute(chunks, profile):
        """Recombine a dict of matrices from chunks_times_tensor to get a full matrix out."""
        def to_list_of_lists(level, partial_position):
            if level == profile.position_len:
                return chunks[tuple(partial_position)]
            return [to_list_of_lists(level + 1, partial_position + [i]) for i in range(profile.deepest_split)]
        return np.block(to_list_of_lists(0, []))

    def size(tree):
        """Calculate the number of floats (or doubles, or complex numbers, or whatever) required to explicitly represent a MultiFactorTree.  The first return value counts just the tensors (which are normally implicit); the second counts both the tensors and the explicit matrix values."""
        matrix_size = sum(prod(matrix.shape) for _index, _row, matrix in tree.triples)
        if tree.children == []:
            tensor_size = tree.subtensor.size
            return tensor_size, tensor_size + matrix_size
        children = (size(child) for child in tree.children)
        tensor_size_only, tensor_size_with_matrix_size = 0, 0
        for tso, tsms in children:
            tensor_size_only += tso
            tensor_size_with_matrix_size += tsms
        return tensor_size_only, tensor_size_with_matrix_size + matrix_size

    def max_rank(tree):
        """Determine the maximum rank at every level of a tree."""
        if tree.children == []:
            return [tree.triples[0][2].shape[1]]
        rank_children = [max_rank(child) for child in tree.children]
        return [tree.triples[0][2].shape[1]] + [max(r[i] for r in rank_children) for i in range(len(rank_children[0]))]

    profile = FactorProfile(dimensions = 6, factor_indices = [0, 1, 2], position_indices = [3, 4, 5], levels = levels)

    tick()
    tree = factor_full(profile)
    time_to_factor = tock()
    
    A = np.random.rand(N, N, N)

    tick()
    chunks = matrix_to_chunks(A, tree, profile)
    tensored_chunks = chunks_times_tensor(chunks, profile)
    compressed_result = reconstitute(tensored_chunks, profile)
    time_to_compress = tock()

    _tsize, total_size = size(tree)

    return time_to_factor, time_to_compress, total_size, max_rank(tree)

import csv
with open('profile_6d.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["N", "Time to factor", "Time to compress", "Total size", "Max rank"])
    logNs = [3, 4, 5]
    for logN in tqdm(logNs):
        N = 2 ** logN
        ttf, ttc, ts, mr = evaluate(N, logN - 2)
        writer.writerow([N, ttf, ttc, ts, str(mr)])
