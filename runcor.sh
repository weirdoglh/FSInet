#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=12

sim=1

# epoch=1
# T=1000
epoch=10
T=2500

Nm=2500

# rm -r ./data/cor*
# rm -r ./plot/cor*

# simulations
if [ $sim -gt 0 ]
then
    
    # for Nf in 0 25 250
    for Nf in 0 25 30 50 100 250
    do
        # for W in -0.1 0.1
        for W in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
        do
            # for B in 0.1
            for B in 0.1 0.3 0.5 0.7 0.9
            do
                for D in 0.0 # 2.0 4.0 6.0 10.0 20.0 40.0 80.0
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $D
                        wait
                        COUNTER=1
                    else
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $D &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
        wait
    done
fi

wait
