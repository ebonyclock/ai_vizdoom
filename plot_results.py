#!/usr/bin/python

import pickle
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.style.use('bmh')

max_len = 104
targets = None
filename = sys.argv[1]
if len(sys.argv) > 2:
    targets = sys.argv[2:]


d = pickle.load(open(filename, "r"))

if targets is not None:
    for target in targets:
        plt.plot(d[target][:max_len], label=target)

else:
    plt.plot(d["max"][:max_len], 'g-', label="max")
    plt.plot(d["mean"][:max_len], 'b-', label="mean")
    plt.plot(d["min"][:max_len], 'r-', label="min")
    plt.plot(d["std"][:max_len], '-', label="std", color="black")
legend = plt.legend(loc='upper left')
for legobj in legend.legendHandles:
    legobj.set_linewidth(3.0)

plt.show()

