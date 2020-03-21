import pandas as pd
import numpy as np
from chart import Chart
from tqdm import tqdm
import itertools

samples = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94]
probabilities = {i: samples[i] for i in range(0, len(samples))}
T = 10

def createChart():
    g = Chart(scheme.shape[0])

    for index, row in scheme.iterrows():

        if list(row[row.apply(lambda x: x == 1)].index):

            for dest in row[row.apply(lambda x: x == 1)].index:
                g.addEdge(index, dest)

    return g


scheme = pd.read_csv('scheme.csv', header=None)

outputs = scheme[~(scheme != 0).any(axis=1)]

result = []
for output in list(outputs.index):
    g = createChart()
    s = 0
    d = output
    g.printAllPaths(s, d)
    result.append(g.getAllPaths())

all_paths = []
for top_list in result:
    for hidden_list in top_list:
        all_paths.append(hidden_list)

unique_all_paths = [list(x) for x in set(tuple(x) for x in all_paths)]

all_comb = []
for lenght in range(1, scheme.shape[0] + 1):
    all_comb.extend(list(itertools.combinations(range(scheme.shape[0]), lenght)))

working_comb = []
for path in tqdm(unique_all_paths):
    for comb in all_comb:
        if set(path).difference(set(comb)) == set():
            working_comb.append(comb)

unique_working_paths = [list(x) for x in set(tuple(x) for x in working_comb)]

result = []
for path in unique_working_paths:
    mult = 1
    for node in path:
        mult *= probabilities.get(node)
    for node in list(set(range(scheme.shape[0])).difference(set(path))):
        mult *= 1 - probabilities.get(node)
    result.append(mult)

p = sum(result)
print("P = " + str(p))

lambd = - np.log(p) / T
print('/\ = ' + str(lambd))

Tndv = 1 / lambd
print('Tndv = ' + str(Tndv))
