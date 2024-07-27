""" A cortical neuron motif simulator
"""

# system
import time
import os

# tools
import argparse
import json
import pandas as pd

# computation libs
import numpy as np

# plot
import matplotlib.pyplot as plt
from lib.mviz import visSpk, visCurve, multiPlot

# simulation libs
import nest

def main():   
    """Interneuron motif PC-FSI-MSN
    """
    # create argument parser
    parser = argparse.ArgumentParser(description='Striatal Microcircuit.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data')
    parser.add_argument('--fpath', type=str, help='data path', default='./plot')
    # simulation setting
    parser.add_argument('--T', type=int, help='simulation time', default=1000)
    parser.add_argument('--Nm', type=int, help='number of MSNs', default=2000)
    parser.add_argument('--Nf', type=int, help='number of FSIs', default=0)
    # input setting
    parser.add_argument('--B', type=float, help='between-pool correlation', default=0.5)
    parser.add_argument('--D', type=float, help='recurrent connection delay', default=2.0)
    parser.add_argument('--S', type=float, help='inhibitory synaptic time constant', default=2.0)
    # neuron setting
    parser.add_argument('--ntype', type=str, help='neuron type', default='iaf_cond_alpha')
    # visualize
    parser.add_argument('--viz', action='store_true', default=False)

    # parsing
    print('Parsing arguments ... ... ')
    # parse argument
    args = parser.parse_args()
    simtime = args.T
    # paths
    suffixpath = '/pls-S%.1f/Nf%d-D%s-B%s/'%(args.S, args.Nf, args.D, args.B)
    recpath = args.dpath + suffixpath                                   # path for saving recordings (gdf files)
    figpath = args.fpath + suffixpath                                   # path for saving figures
    spkpath = figpath + 'spk/'
    ratpath = figpath + 'rat/'
    wtspath = figpath + 'wt/'
    for path in [recpath, spkpath, ratpath, wtspath]:
        os.makedirs(path, exist_ok=True)

    #* Motif network
    # neurons
    nTypes = ['MSN', 'FSI']
    nNums = [args.Nm, args.Nf]
    nparam = {'C_m': 250., 'E_L': -70., 'g_L': 50/3, 'V_th': -55., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in':args.S}
    nParams = [nparam, nparam]

    # connection degree
    MSN_deg = int(nNums[0]/10)
    FSI_deg = int(nNums[0]/100)

    # input
    print('inputs')
    B = args.B
    nstm = 100

    # delays
    if B > 0:
        delays = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 40.0])
        # delays = np.array([0.0, 1.0, 10.0])
    else:
        delays = np.array([0.0])
    epoch = len(delays)
    print(epoch)

    # response population size
    MSN_num = int(nNums[0]/10/2)

    # fixed cortical inputs
    msd = int(time.time() * 1000.0) % 4294967295
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    pulse_times = []
    if B > 0:
        # spike trains
        pulses = np.sort(np.concatenate([[250.0], np.random.uniform(250.0, simtime-250.0, (simtime-500)//200-1)]))
        for e in range(epoch):
            pulse_times.append(pulses + simtime*e)
        ppg_pars = {"pulse_times": np.concatenate(pulse_times), "activity": 30, "sdev": 5.0}
        stm = nest.Create("pulsepacket_generator", 1, ppg_pars)

        # input recording
        inDetector = nest.Create('spike_recorder', 1)
        nest.Connect(stm, inDetector)
        # run simulation
        nest.Simulate(simtime*epoch)
        # get spike trains
        inSpks = nest.GetStatus(inDetector, 'events')[0]
        inTimes, inIds = inSpks['times'], np.random.randint(0, int(nstm*(2-B)), len(inSpks['senders']))
        # plot
        visSpk(inTimes, inIds, path=spkpath + 'input')

    # connection strengths
    Js = [  [1.3, 1.05],
            [1.7, 0.4],
            [0.3, 0.],
            [2.0, 0.]]

    # reset kernel
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    # neurons
    print('create neurons')
    nPops = {}
    nPops['MSN'] = nest.Create('iaf_cond_alpha', nNums[0])
    nest.SetStatus(nPops['MSN'], nParams[0])
    if nNums[-1] > 0:
        nPops['FSI'] = nest.Create('iaf_cond_alpha', nNums[-1])
        nest.SetStatus(nPops['FSI'], nParams[-1])

    # background noise
    if args.Nf > 0:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 7.2e3})
        Js[1][0] = 3.0
    else:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 5.8e3})
        Js[1][0] = 1.8
    bkgF = nest.Create('poisson_generator', 1, {'rate': 6.9e3})
    if B > 0:
        # cortical input
        stm = nest.Create('spike_generator', nstm*2)
        # input for population A
        for i in range(nstm):
            stm[i].spike_times = inTimes[inIds == i]
        # input for population B
        for i in range(nstm):
            stm[-i].spike_times = inTimes[inIds == int(i + nstm*(1-B))]

    # connections
    print('generating connections')
    # background input (excitatory)
    nest.Connect(bkgM, nPops['MSN'], syn_spec={'weight': Js[0][0], 'delay': 1.0})
    if args.Nf > 0:
        nest.Connect(bkgF, nPops['FSI'], syn_spec={'weight': Js[0][-1], 'delay': 1.0})
    # cortical stimulus (excitatory)
    if B > 0:
        syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][0], 'delay': 2.0}
        nest.Connect(stm[:nstm], nPops['MSN'][nNums[0]//2-MSN_num:nNums[0]//2], syn_spec=syn_spec)
        nest.Connect(stm[-nstm:], nPops['MSN'][nNums[0]//2:nNums[0]//2+MSN_num], syn_spec=syn_spec)
        if args.Nf > 0:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][-1], 'delay': 1.0}
            nest.Connect(stm, nPops['FSI'], syn_spec=syn_spec)
    # strial connection
    con_spec = {'rule': 'fixed_indegree', 'indegree': MSN_deg, 'allow_autapses': False}
    syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[2][0], 'delay': args.D}
    nest.Connect(nPops['MSN'], nPops['MSN'], con_spec, syn_spec)
    if args.Nf > 0:
        con_spec = {'rule': 'fixed_indegree', 'indegree': FSI_deg}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[-1][0], 'delay': 1.0}
        nest.Connect(nPops['FSI'], nPops['MSN'], con_spec, syn_spec)

    # record device
    print('link devices')
    # 'record_to': 'ascii'
    spikeDetector = nest.Create('spike_recorder', 1, {'label':recpath + 'spk'})
    for npop in nPops.values():
        nest.Connect(npop, spikeDetector)
    if B > 0:
        inDetector = nest.Create('spike_recorder', 1, {'label':recpath + 'pulse'})
        nest.Connect(stm, inDetector)
    
    # subthreshold recording
    if args.Nf == 0:
        del nTypes[-1]
        del nNums[-1]
    subNeurons = [nPops[ntype][0] for ntype in nTypes]
    mulDetects = [nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': 0.1}) for ntype in nTypes]
    for nrn, det in zip(subNeurons, mulDetects):
        nest.Connect(det, nrn)

    # simulation
    print('simulating')
    initstates = [np.random.uniform(-70., -55., len(npop)) for npop in nPops.values()]
    for e in range(epoch):
        # initial state
        for npop, rvs in zip(nPops.values(), initstates):
            nest.SetStatus(npop, params='V_m', val=rvs.tolist())

        if B > 0:
            # set delay
            for i in range(nstm):
                stm[i].spike_times = inTimes[inIds == i] + delays[e]

        # simulate
        nest.Simulate(simtime)

    # visualize
    print('visualize')
    if B > 0:
        # plot input pulse packets
        pulseEvents = nest.GetStatus(inDetector, 'events')[0]
        pulseTimes, pulseIds = pulseEvents['times'], pulseEvents['senders']
        visSpk(pulseTimes, pulseIds-nNums[0], path=spkpath + 'pulse')

    # spikes
    visnum = 200
    spikeEvents = nest.GetStatus(spikeDetector, 'events')[0]
    spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']
    mask = (spikeIds > (nNums[0]//2)-visnum) & (spikeIds <= (nNums[0]//2) + visnum)
    visSpk(spikeTimes[mask], spikeIds[mask]-(nNums[0]//2)+visnum, path=spkpath + 'spk')

    # multimeter
    multiEvents = [nest.GetStatus(detector)[0]['events'] for detector in mulDetects]
    multiPlot(multiEvents, [0., simtime], nTypes, ratpath + 'vg_')

    # firing rates
    nNums = [nNums[0]//2, nNums[0]//2, args.Nf]
    nTypes = ['M1', 'M2', 'FSI']
    if args.Nf == 0:
        del nNums[-1]
        del nTypes[-1]
    binsize = 100
    tmBins = np.arange(0, simtime*epoch+1, binsize)
    idBins = np.concatenate([[0], np.cumsum(nNums)]) + 1
    ratesPop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0] * 1e3 / binsize / nNums
    visCurve([tmBins[:-1]]*len(nTypes), ratesPop.T,
             'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_', nTypes)
    print('mean firing rate: ', np.mean(ratesPop, axis=0))

    # save recording
    np.savez(recpath + 'spk.npz', spikeIds, spikeTimes, pulse_times)    
    print('done')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))