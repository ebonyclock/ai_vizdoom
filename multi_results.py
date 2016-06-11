#!/usr/bin/python

import pickle
import sys
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.style.use('ggplot')

targets = None
filename = sys.argv[1]

for target in sys.argv[1:]:
    d = pickle.load(open(target, "r"))
    plt.plot(d["mean"], label=target)

legend = plt.legend(loc='lower right',fancybox=True, shadow=True,prop={'size':10}).draggable()
#for legobj in legend.legendHandles:
    #legobj.set_linewidth(3.0)



plt.show()
#   plt.savefig("health_supreme.png")
