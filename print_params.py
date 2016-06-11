#!/usr/bin/python

import pickle
import sys

for filename in sys.argv[1:]:
    params = pickle.load(open(filename, "r"))[0]
    for key in params:

        if key =="network_args":
            print "network_args:"
            for net_key in params[key]:
                print "\t",net_key, params[key][net_key]
        else:
            print key, ":", params[key]
