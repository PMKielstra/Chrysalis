import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

import numpy as np
from matplotlib import pyplot as plt

from factorprofile import Profile, BOTH, UP, DOWN
from factor import build_factor_forest
from multiwaymatvec import apply
from tensor import AK_true
from profiling import ss_accuracy, total_memory, max_leaf_row_length_forests, evaluate_top_translation_invariance

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
parser.add_argument("--distance", type=float, default=1)
parser.add_argument("--asMatrix", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--matvec", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--down", action="store_true")
parser.add_argument("--translationInvariant", action="store_true")
parser.add_argument("--flat", action="store_true")
args = parser.parse_args()

PoolExecutor = ProcessPoolExecutor #MPIExecutor if args.mpi else ProcessPoolExecutor

with PoolExecutor() as pool:
##    if args.mpi:
##        pool.workers_exit()
    logNs = [int(args.logN)]
    for logN in logNs:
        N = 2 ** (logN + 3)
        profile = Profile(
            N = N,
            dimens = int(args.dimens),
            eps = eps,
            levels = (logN if args.down else ceil(logN / 2)),
            direction = (DOWN if args.down else BOTH),
            subsamples = 30,
            as_matrix = args.asMatrix,
            verbose = args.verbose,
            distance = args.distance,
            translation_invariant = args.translationInvariant,
            use_fake = not (args.matvec or args.accuracy),
            flat = args.flat
            )
        print(f"N: {N}")
        if args.flat:
            print("Flat")
        print(f"Distance: {args.distance}")
        tick()
        factor_forests = []
        for axis in range(profile.dimens):
            profile.set_axis_roll(axis)
            factor_forests.append(build_factor_forest(pool, profile))
        profile.set_axis_roll(0)
        ttf = tock()
        ts = total_memory(profile, factor_forests)[0]
        mr = max_leaf_row_length_forests(factor_forests)
        tti = evaluate_top_translation_invariance(profile, factor_forests)
        print(f"Time to factor: {ttf}")
        print(f"Total memory: {ts}")
        print(f"Top translation invariance: {tti}")
        print(f"Max rows at leaf level: {mr}", flush=True)
        if args.matvec or args.accuracy:
            A = np.random.rand(* [profile.N] * profile.dimens)
            tick()
            compressed_AK = apply(profile, A, factor_forests)
            ttc = tock()
            print(f"Time to apply: {ttc}", flush=True)
            if args.accuracy:
                accuracy = ss_accuracy(profile, A, compressed_AK)
                print(f"Accuracy: {accuracy}", flush=True)
