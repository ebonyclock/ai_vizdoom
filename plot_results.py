#!/usr/bin/python

import pickle
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.style.use('ggplot')

targets = None
filename = sys.argv[1]
if len(sys.argv) > 2:
    targets = sys.argv[2:]


d = pickle.load(open(filename, "r"))

if targets is not None:
    for target in targets:
        plt.plot(d[target], label=target)

else:
    plt.plot(d["max"], 'go-', label="max")
    plt.plot(d["mean"], 'bo-', label="mean")
    plt.plot(d["min"], 'ro-', label="min")
    plt.plot(d["std"], '-', label="std", color="black")
legend = plt.legend(loc='upper left',fancybox=True, shadow=True).draggable()



plt.show()
#plt.savefig("defend_the_line.pdf")

