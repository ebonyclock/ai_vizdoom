#!/usr/bin/python

import pickle
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.style.use('ggplot')

targets = None
filename = sys.argv[1]

for target in sys.argv[1:]:
    d = pickle.load(open(target, "r"))
    if "actions" in d:
        mil_steps = np.array(d["actions"]) / 1000000.0
        plt.plot(mil_steps, d["mean"], label=target)
    else:
        plt.plot(d["mean"], label=target)

legend = plt.legend(loc='lower right', fancybox=True, shadow=True, prop={'size': 10}).draggable()
plt.xlabel('10^6 actions')
# for legobj in legend.legendHandles:
# legobj.set_linewidth(3.0)



plt.show()
#   plt.savefig("health_supreme.png")
