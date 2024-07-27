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
    """Interneuron motif PC-PV-SOM-VIP
    """
    # create argument parser
    parser = argparse.ArgumentParser(description='Interneuron motif PC-PV-SOM-VIP.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data')
    parser.add_argument('--fpath', type=str, help='data path', default='./plot')
    # simulation setting
    parser.add_argument('--epoch', type=int, help='number of epochs', default=2)
    parser.add_argument('--T', type=int, help='simulation time', default=1000)
    parser.add_argument('--Nm', type=int, help='number of MSNs', default=5000)
    parser.add_argument('--Nf', type=int, help='number of FSIs', default=0)
    # input setting
    parser.add_argument('--W', type=float, help='within-pool correlation', default=0.01)
    parser.add_argument('--D', type=float, help='recurrent connection delay', default=2.0)
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
    suffixpath = '/var-D%.1f/Nf%d-W%s/'%(args.D, args.Nf, args.W)
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
    # nparam = {'C_m': 250., 'E_L': -70., 'g_L': 50/3, 'V_th': -55., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in':2.0}
    nparam = {'C_m': 250., 'E_L': -70., 'g_L': 50/3, 'V_th': -55., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in':15.0}
    nParams = [nparam, nparam]

    MSN_deg = int(nNums[0]/10)
    # FSI_deg = int(nNums[0]/100)
    FSI_deg = int(nNums[0]/100*0.6)
    CTX_deg = 100

    # input
    print('inputs')
    W = args.W

    # fixed cortical inputs
    msd = int(time.time() * 1000.0) % 4294967295
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    if W > 0:
        nstm = int(CTX_deg / W)
        # spike trains
        stm = nest.Create('poisson_generator', nstm, params={'rate': 10})
        # input recording
        inDetector = nest.Create('spike_recorder', 1)
        nest.Connect(stm, inDetector)
        # run simulation
        nest.Simulate(simtime)
        # get spike trains
        inSpks = nest.GetStatus(inDetector, 'events')[0]

    # # connection strengths
    # Js = [  [1.1, 1.1],
    #         [2.1, 0.7],
    #         [0.4, 0.],
    #         [0.8, 0.]]
    
    # very strong FSI inhibition compared to MSN
    Js = [  [1.0, 1.0],
            [6.5, 0.7],
            [0.1, 0.],
            [0.5, 0.]]

    # reset kernel
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    # neurons
    print('create neurons')
    nPops = {}
    posMSN = nest.spatial.free(nest.random.uniform(min=-1, max=1), num_dimensions=2)
    nPops['MSN'] = nest.Create('iaf_cond_alpha', nNums[0], positions = posMSN)
    nest.SetStatus(nPops['MSN'], nParams[0])
    if nNums[-1] > 0:
        nPops['FSI'] = nest.Create('iaf_cond_alpha', nNums[-1])
        nest.SetStatus(nPops['FSI'], nParams[-1])

    # # background noise
    # if args.Nf > 0:
    #     bkgM = nest.Create('poisson_generator', 1, {'rate': 7.2e3})
    # else:
    #     bkgM = nest.Create('poisson_generator', 1, {'rate': 6.5e3})
    # bkgF = nest.Create('poisson_generator', 1, {'rate': 6.9e3})
    # background noise
    bkgF = nest.Create('poisson_generator', 1, {'rate': 7.4e3})
    bkgM = nest.Create('poisson_generator', 1, {'rate': 8.4e3})
    # cortical input
    if W > 0:
        stm = nest.Create('spike_generator', nstm)
        for i in range(nstm):
            spks = inSpks['times'][inSpks['senders'] == (i + 1)]
            stm[i].spike_times = np.concatenate([spks + e*simtime for e in range(args.epoch)])

    # connections
    print('generating connections')
    # background input (excitatory)
    nest.Connect(bkgM, nPops['MSN'], 'all_to_all', syn_spec={'weight': Js[0][0], 'delay': 1.0})
    if args.Nf > 0:
        nest.Connect(bkgF, nPops['FSI'], 'all_to_all', syn_spec={'weight': Js[0][-1], 'delay': 1.0})
    # cortical stimulus (excitatory)
    if W > 0:
        con_spec = {'rule': 'fixed_indegree', 'indegree': CTX_deg}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][0], 'delay': 2.0}
        nest.Connect(stm, nPops['MSN'], con_spec, syn_spec)
        if args.Nf > 0:
            con_spec = {'rule': 'fixed_indegree', 'indegree': CTX_deg}
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][-1], 'delay': 1.0}
            nest.Connect(stm, nPops['FSI'], con_spec, syn_spec)
    # recurrent connection (inhibitory)
    con_spec = {'rule': 'fixed_indegree', 'indegree': MSN_deg, 'allow_autapses': False}
    syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[2][0], 'delay': args.D}
    nest.Connect(nPops['MSN'], nPops['MSN'], con_spec, syn_spec)
    # feedforward inhibition
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
    for e in range(args.epoch):
        # initial state
        for npop, rvs in zip(nPops.values(), initstates):
            nest.SetStatus(npop, params='V_m', val=rvs.tolist())

        # simulate
        nest.Simulate(simtime)

    # spikes
    spikeEvents = nest.GetStatus(spikeDetector, 'events')[0]
    spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']

    if args.viz:
        # visualize
        print('visualize')
        mask = (spikeIds > nNums[0]-100) & (spikeIds <= nNums[0] + 100)
        visSpk(spikeTimes[mask], spikeIds[mask]-nNums[0]+100, path=spkpath + 'spk')

        # firing rates
        idBins = np.concatenate([[0], np.cumsum(nNums)]) + 1
        tmBins = np.arange(0, simtime*args.epoch+1, 100)
        rateTimePop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0]*10/nNums
        visCurve([tmBins[:-1]]*len(nTypes), rateTimePop.T,
                'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_', nTypes)
        print(suffixpath)
        print('mean firing rate: ', np.mean(rateTimePop, axis=0))

        # multimeter
        multiEvents = [nest.GetStatus(detector)[0]['events'] for detector in mulDetects]
        multiPlot(multiEvents, [0., simtime], nTypes, ratpath + 'vg_')

    np.savez(recpath + 'spk.npz', spikeIds, spikeTimes)
    print('done')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))