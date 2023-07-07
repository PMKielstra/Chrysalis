import time
import numpy as np
from tqdm import tqdm
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

dimens = 3 # Number of source or observer dimens, not source + observer dimens
eps = 1e-6

import csv
with open(f"profile_{dimens}d.csv", mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["N", "Time to factor", "Time to compress", "Total size", "Max rank"])
    logNs = [4]
    for logN in tqdm(logNs):
        N = 2 ** (logN)
        A = np.random.rand(* [N] * dimens)
        profile = Profile(
            N = N,
            dimens = dimens,
            eps = eps,
            levels = logN // 2,
            direction = BOTH
            )
        tick()
        factor_forest = build_factor_forest(profile)
        ttf = tock()
        ts = total_memory(profile, factor_forest)[0]
        mr = max_leaf_row_length_forest(factor_forest)
        tick()
        apply(A, profile, factor_forest)
        ttc = tock()
        writer.writerow([N, ttf, ttc, ts, mr])
