import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from profile import Profile, BOTH, UP, DOWN
from factor import build_factor_forest
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
parser.add_argument("--matvec", action="store_true")
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
        print(f"N: {N}")
        tick()
        factor_forest = build_factor_forest(pool, profile)
        ttf = tock()
        ts = total_memory(profile, factor_forest)[0]
        mr = max_leaf_row_length_forest(factor_forest)
        print(f"Time to factor: {ttf}")
        print(f"Total memory: {ts}")
        print(f"Max rows at leaf level: {mr}", flush=True)
        if args.matvec:
            tick()
            compressed_AK = apply(profile, A, factor_forest)
            ttc = tock()
        print(f"Time to apply: {ttc}", flush=True)
        if args.accuracy:
            accuracy = ss_accuracy(profile, A, compressed_AK)
            print(f"Accuracy: {accuracy}", flush=True)
