#!/usr/bin/python

import math
import pickle

import numpy as np
from skimage import io
from matplotlib import pyplot as  plt
import sys

params_file = sys.argv[1]
params = pickle.load(open(params_file, "r"))[1]


def concat(a):
    im = np.concatenate(a[0], axis=0)
    for i in range(1, len(a)):
        im = np.concatenate([im, np.concatenate(a[i], axis=0)], axis=1)
    return im


for p in params:
    if len(p.shape) != 4 or p.shape[2] != p.shape[3]:
        continue
    print p.shape
    im = concat(p)

    im -= im.min()
    im /= im.max()
    io.imshow(im)
    io.show()
