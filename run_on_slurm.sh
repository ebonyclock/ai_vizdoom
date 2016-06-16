#!/usr/bin/env bash

for i in  0 1 2 3 4 ; do
    name="health_baseline_duel_${i}"
    jobname="hb4_${i}"
    logfile="logs/${name}"
    export THEANO_FLAGS=base_compiledir=".theano_stuff/duel_${i}"
    srun  -J "$jobname" -x ./excluded_hosts -p lab-ci --constraint=cuda --exclusive  python -u learn.py health_baseline -n "$name" --no-save --no-save-best --no-tqdm -e 25 > "$logfile"  &
done

