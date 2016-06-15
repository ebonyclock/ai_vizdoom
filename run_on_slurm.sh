#!/usr/bin/env bash

instances=5
for i in  $(seq 0 $((instances-1))) ; do
    name=health_baseline_$i
    logfile=logs/$name
    export THEANO_FLAGS=base_compiledir=.theano_stuff/$i
    srun  -x ./excluded_hosts -p lab-ci --constraint=cuda --exclusive  python -u learn.py health_baseline -n $name --no-tqdm -e 25 > $logfile  &
done

