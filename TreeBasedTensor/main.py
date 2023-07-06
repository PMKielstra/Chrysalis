import time
import numpy as np
from tqdm import tqdm
from factor import build_factor_forest, UP, DOWN, BOTH
from matvec import apply
from tensor import AK_true
from profiling import ss_accuracy, total_memory, max_leaf_row_length

eps = 1e-6

t = 0
def tick():
    global t
    t = time.time()

def tock():
    global t
    return time.time() - t

import csv
with open('profile.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["N", "Time to factor", "Time to compress", "Total size", "Max rank"])
    logNs = [2, 4]
    for logN in tqdm(logNs):
        N = 2 ** (logN + 3)
        A = np.random.rand(N, N)
        tick()
        factor_forest = build_factor_forest(N, eps, logN // 2, BOTH)
        ttf = tock()
        ts = total_memory(factor_forest)[0]
        tick()
        apply(A, factor_forest)
        ttc = tock()
        N, levels, off_cols_lists, trees, direction = factor_forest
        mr = max(max_leaf_row_length(t) for t in trees)
        writer.writerow([N, ttf, ttc, ts, mr])
