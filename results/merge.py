#!/usr/bin/python

import pickle
import sys

targets = None
filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = sys.argv[3]

max_len = 120
a = pickle.load(open(filename1, "r"))
b = pickle.load(open(filename2, "r"))
res = dict()

for k in a.keys():
    aa = a[k]
    bb = b[k]
    if k=="epoch":
        res[k] = range(len(aa)+len(bb))[0:max_len]
        continue


    if k=="overall_time":
        for i in range(len(bb)):
            bb[i] += aa[-1]
    r = (a[k] + b[k])[0:max_len]
    res[k] = r

pickle.dump(res, open(filename3, "w"))