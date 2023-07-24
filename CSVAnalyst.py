import numpy as np
import csv
from matplotlib import pyplot as plt

csv_data = []
with open("Perlmutter data/Perlmutter_2d.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        csv_data.append([float(i) if len(i) > 0 else None for i in row])

Ns = np.array([d[0] for d in csv_data])

def plotloglog(data_indices, data_titles, title, yaxis, orders):
    data_streams = [[d[data_index] for d in csv_data] for data_index in data_indices]
    for data, data_title in zip(data_streams, data_titles):
        plt.loglog([Ns[i] for i in range(len(data)) if data[i] is not None], list(filter(lambda x: x is not None, data)), label=data_title)
    for function, label, index in orders:
        data = data_streams[index]
        final = data[-1]
        data = data[:-1]
        while final is None:
            final = data[-1]
            data = data[:-1]
        plt.loglog(Ns, 1.3 * function(Ns) * final / function(Ns[-1]), label=label)
    plt.xlabel("N")
    plt.ylabel(yaxis)
    plt.title(title)
    plt.legend()
    plt.show()


indices = np.array((1, 2, 3))
titles = ("Tensor hybrid", "Tensor downward", "Matrix hybrid")
plotloglog(indices, titles, "Memory used to factor a tensor of size N^4", "Memory (complex float64s)", [(lambda n: n ** 2, "O(N^2)", 0), (lambda n: n ** 2 * (np.log(n)), "O(N^2 log N)", 2)])
plotloglog(indices + 3, titles, "Time to factor a tensor of size N^4, with 128 cores", "Time (seconds)", [(lambda n: n ** 2, "O(N^2)", 0)])
plotloglog(indices + 6, titles, "Number of rows at tensor-computation level for a tensor of size N^4", "", [])
