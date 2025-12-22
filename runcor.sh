#!/bin/bash

# number of thread the computer supports to run in parallel
COUNTER=1
THREAD=50

sim=0
Nm=2500

if [ $sim == 0 ]
then
epoch=10
T=2500
    for delay in 2.0
    do
        for Nf in 0 25 250
        do
            for W in -0.1 0.1
            do
                for B in 0.1 0.9
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay --viz
                        wait
                        COUNTER=1
                    else
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay --viz &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi
wait


# simulations
if [ $sim == 1 ]
then
epoch=100
T=2500
    for delay in 2.0
    do
        for Nf in 0 25 30 50 90 150 250
        do
            for W in -0.1 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
            do
                for B in 0.1 0.3 0.5 0.7 0.9
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay
                        wait
                        COUNTER=1
                    else
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi 

if [ $sim == 2 ]
then
epoch=10
T=2500
    for delay in 2.0
    do
        for Nf in 25 250
        do
            for W in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
            do
                for B in 0.1 0.5 0.9
                do
                    echo $COUNTER
                    if [ $COUNTER == $THREAD ]
                    then
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay
                        wait
                        COUNTER=1
                    else
                        python runcor.py --Nm $Nm --Nf $Nf --W $W --B $B --epoch $epoch --T $T --D $delay &
                        COUNTER=$(( COUNTER + 1 ))
                    fi
                done
            done
        done
    done
fi 

wait
