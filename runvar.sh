#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=0
THREAD=5

sim=1

epoch=10
T=2500

rm -r ./data/var*
rm -r ./plot/var*

# simulations
if [ $sim -gt 0 ]
then
    for delay in 1.0 2.0 3.0
    do
        for Nf in 0 20 40 60 80 100 120 140
        do
            for W in 0.01 0.02 0.04 0.06 0.08 0.1 0.15 0.2 0.25
            do
                echo $COUNTER
                if [ $COUNTER == $THREAD ]
                then
                    python runvar.py --Nf $Nf --W $W --epoch $epoch --T $T --D $delay
                    wait
                    COUNTER=0
                else
                    python runvar.py --Nf $Nf --W $W --epoch $epoch --T $T --D $delay &
                    COUNTER=$(( COUNTER + 1 ))
                fi
                # exit 1
            done
        done
    done
fi
