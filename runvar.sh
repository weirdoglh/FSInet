#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=6

sim=1

# epoch=1
# T=1000
epoch=10
T=2500

Nm=2500

rm -r ./data/var*
rm -r ./plot/var*

# simulations
if [ $sim -gt 0 ]
then
    for delay in 2.0
    do
        # for Nf in 0 25 250
        for Nf in 0 25 30 50 90 150 250
        do
            # for W in -0.1 0.1
            for W in 0.01 0.025 0.05 0.075 0.1 0.2 0.3 0.4 0.5
            do
                echo $COUNTER
                if [ $COUNTER == $THREAD ]
                then
                    python runvar.py --Nm $Nm --Nf $Nf --W $W --epoch $epoch --T $T --D $delay
                    wait
                    COUNTER=1
                else
                    python runvar.py --Nm $Nm --Nf $Nf --W $W --epoch $epoch --T $T --D $delay &
                    COUNTER=$(( COUNTER + 1 ))
                fi
                # exit 1
            done
        done
    done
fi
wait
