#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=6

sim=1

T=2500

Nm=2000

rm -r ./data/pls*
rm -r ./plot/pls*

# simulations
if [ $sim -gt 0 ]
then
    # for S in 10.0
    for S in 2.0 5.0 10.0 15.0 20.0
    do
        # for delay in 1.0
        for delay in 1.0 2.0 3.0
        do
            # for Nf in 0 20 200
            for Nf in 0 20 50 100 200
            do
                # for B in -0.1 0.5
                for B in 0.1 0.3 0.5 0.7 0.9
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runpls.py --Nm $Nm --Nf $Nf --S $S --B $B --T $T --D $delay
                        wait
                        COUNTER=1
                    else
                        python runpls.py --Nm $Nm --Nf $Nf --S $S --B $B --T $T --D $delay &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi

wait
