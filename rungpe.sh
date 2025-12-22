#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=45

rate=1.0e4

# simulations
for Nf in 0 25 30 50 90 150 250
do
    for W in -0.1 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
    do
        for B in 0.1 0.3 0.5 0.7 0.9
        do
            echo $COUNTER
            if [ $COUNTER == $THREAD ]
            then
                python rungpe.py --Nf $Nf --W $W --B $B --rate $rate
                wait
                COUNTER=1
            else
                python rungpe.py --Nf $Nf --W $W --B $B --rate $rate &
                COUNTER=$(( COUNTER + 1 ))
            fi
        done
    done
done
wait
