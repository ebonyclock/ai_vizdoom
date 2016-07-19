#!/usr/bin/env bash

for i in  0 1 2  3 4 5 6 7 8 9 ; do
    basename="cover_allhot_skip3"
    config="config/take_cover.cfg"
    agent="agents/var_baseline4.json"

    export THEANO_FLAGS=base_compiledir=".theano_stuff/hb${i}"

    name="${basename}_run${i}"
    jobname="${i}_${basename}"
    logfile="logs/${name}"
    # to exclude hosts: -x $HOSTFILE
    srun   -x ./excluded_hosts --exclusive -J "$jobname"  -p lab-ci --constraint=cuda python -u learn.py -j "$agent" -n "$name" -c "$config" --no-save --no-tqdm -e 25 > "$logfile"  &
done


