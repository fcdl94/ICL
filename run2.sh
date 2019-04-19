#!/usr/bin/env bash

torun=2
lr=0.01

for mem in 100 2000 50000; do
    for (( i=1; i<=$1; i++ )); do
        echo " "
        echo "Running "$mem ${i}
        python main.py imnistm -m icarl --lr ${lr} --mem_size ${mem} -l icarl-m${mem}-l${lr}_${i} --to_run ${torun}
    done
done
