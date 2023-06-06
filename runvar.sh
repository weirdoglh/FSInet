#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=5

sim=1

epoch=1
T=1000
# epoch=10
# T=2500

rm -r ./data/var*
rm -r ./plot/var*

# simulations
if [ $sim -gt 0 ]
then
    for delay in 3.0
    do
        for Nf in 0 100
        # for Nf in 0 100 200 300 400 600 800 1000
        do
            for W in -0.01 0.01
            # for W in -0.01 0.01 0.02 0.04 0.06 0.08 0.1 0.15 0.2 0.25
            do
                echo $COUNTER
                if [ $COUNTER == $THREAD ]
                then
                    python runvar.py --Nf $Nf --W $W --epoch $epoch --T $T --D $delay
                    wait
                    COUNTER=1
                else
                    python runvar.py --Nf $Nf --W $W --epoch $epoch --T $T --D $delay &
                    COUNTER=$(( COUNTER + 1 ))
                fi
                # exit 1
            done
        done
    done
fi
wait
