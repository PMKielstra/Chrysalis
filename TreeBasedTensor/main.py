import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from factor import Profile, build_factor_forest, UP, DOWN, BOTH
from matvec import apply
from tensor import AK_true
from profiling import ss_accuracy, total_memory, max_leaf_row_length_forest

t = 0
def tick():
    global t
    t = time.time()

def tock():
    global t
    return time.time() - t

eps = 1e-6

parser = argparse.ArgumentParser()
#parser.add_argument("--mpi", action="store_true")
parser.add_argument("--logN")
parser.add_argument("--dimens") # Number of source or observer dimens, not source + observer dimens
parser.add_argument("--asMatrix", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

PoolExecutor = ProcessPoolExecutor #MPIExecutor if args.mpi else ProcessPoolExecutor

with PoolExecutor() as pool:
##    if args.mpi:
##        pool.workers_exit()
    logNs = [int(args.logN)]
    for logN in logNs:
        N = 2 ** (logN + 3)
        A = np.random.rand(* [N] * int(args.dimens))
        profile = Profile(
            N = N,
            dimens = int(args.dimens),
            eps = eps,
            levels = logN // 2,
            direction = BOTH,
            subsamples = 20,
            as_matrix = args.asMatrix,
            verbose = args.verbose
            )
        tick()
        factor_forest = build_factor_forest(pool, profile)
        ttf = tock()
        ts = total_memory(profile, factor_forest)[0]
        mr = max_leaf_row_length_forest(factor_forest)
        tick()
        compressed_AK = apply(A, profile, factor_forest)
        ttc = tock()
        if args.accuracy:
            accuracy = ss_accuracy(profile, A, compressed_AK)
        else:
            accuracy = -1
        print(N, ttf, ttc, ts, mr, accuracy)
