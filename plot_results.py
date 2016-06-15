#!/usr/bin/python

import pickle
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.style.use('ggplot')

targets = None
filename = sys.argv[1]
if len(sys.argv) > 2:
    targets = sys.argv[2:]

d = pickle.load(open(filename, "r"))

if targets is not None:
    for target in targets:
        mil_steps = np.array(d["actions"]) / 1000000.0
        plt.plot(mil_steps,d[target], label=target)

else:
    mil_steps = np.array(d["actions"]) / 1000000.0
    plt.plot(mil_steps, d["max"], 'g-', label="max")
    plt.plot(mil_steps, d["mean"], 'b-', label="mean")
    plt.plot(mil_steps, d["min"], 'r-', label="min")
    plt.plot(mil_steps, d["std"], '-', label="std", color="black")
plt.xlabel('10^6 actions')
legend = plt.legend(loc='upper left', fancybox=True, shadow=True).draggable()

plt.show()
# plt.savefig("defend_the_line.pdf")
