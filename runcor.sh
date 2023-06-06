#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=6

sim=1

epoch=10
T=2500

rm -r ./data/cor*
rm -r ./plot/cor*

# simulations
if [ $sim -gt 0 ]
then
    # for delay in 1.0 2.0 3.0
    for delay in 3.0
    do
        # for Nf in 0 100
        for Nf in 0 100 200 300 400 600 800 1000
        do
            for W in 0.01 0.05 0.1 0.15 0.2 0.25
            do
                # for B in 0.1
                for B in 0.1 0.2 0.3 0.5 0.7 0.9
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runcor.py --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay
                        wait
                        COUNTER=1
                    else
                        python runcor.py --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi
