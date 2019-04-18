#!/usr/bin/env bash

torun=2
lr=2

for mem in 100 2000 50000; do
    for (( i=1; i<=$1; i++ )); do
        echo " "
        echo "Running "$mem ${i}
        python main.py isvhn -m icarl --lr ${lr} --mem_size ${mem} -l icarl-m${mem}-l${lr}_${i} --epochs 150 --to_run ${torun}
    done
done
