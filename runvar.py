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
    parser.add_argument('--Nm', type=int, help='number of MSNs', default=10000)
    parser.add_argument('--Nf', type=int, help='number of FSIs', default=100)
    # input setting
    parser.add_argument('--W', type=float, help='within-pool correlation', default=0.01)
    parser.add_argument('--D', type=float, help='recurrent connection delay', default=3.0)
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
    suffixpath = '/var-D%.1f/Nf%d-W%.2f/'%(args.D, args.Nf, args.W)
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
    nparam = {'C_m': 250., 'E_L': -70., 'g_L': 50/3, 'V_th': -55., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in':2.0}
    nParams = [nparam, nparam]

    # input
    print('inputs')
    W = args.W

    # fixed cortical inputs
    msd = int(time.time() * 1000.0) % 4294967295
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)
    # spike trains
    mip = nest.Create('mip_generator', params={'rate': 40./np.abs(W), 'p_copy': np.abs(W)})
    stm = nest.Create('parrot_neuron', 1000)
    nest.Connect(mip, stm)
    # input recording
    inDetector = nest.Create('spike_recorder', 1)
    nest.Connect(stm, inDetector)
    # run simulation
    nest.Simulate(simtime)
    # get spike trains
    inSpks = nest.GetStatus(inDetector, 'events')[0]

    # connection strengths
    Js = [  [1.1, 1.1],
            [1.8, 0.8],
            [0.3, 0.],
            [0.2, 0.]]

    # reset kernel
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)
    # background noise
    if args.Nf > 0:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 7.0e3})
    else:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 6.2e3})
    bkgF = nest.Create('poisson_generator', 1, {'rate': 7.0e3})
    # cortical input
    stm = nest.Create('spike_generator', 1000)
    for i in range(1000):
        spks = inSpks['times'][inSpks['senders'] == (i + 1)]
        stm[i].spike_times = np.concatenate([spks + e*simtime for e in range(args.epoch)])

    # neurons
    print('create neurons')
    nPops = {}
    nPops['MSN'] = nest.Create('iaf_cond_alpha', nNums[0])
    nest.SetStatus(nPops['MSN'], nParams[0])
    if nNums[-1] > 0:
        nPops['FSI'] = nest.Create('iaf_cond_alpha', nNums[-1])
        nest.SetStatus(nPops['FSI'], nParams[-1])

    # connections
    print('generating connections')
    # background input (excitatory)
    nest.Connect(bkgM, nPops['MSN'], 'all_to_all', syn_spec={'weight': Js[0][0]})
    if args.Nf > 0:
        nest.Connect(bkgF, nPops['FSI'], 'all_to_all', syn_spec={'weight': Js[0][-1]})

    # striatal connection (inhibitory)
    con_spec = {'rule': 'fixed_indegree', 'indegree': 50}
    syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[2][0], 'delay': args.D}
    nest.Connect(nPops['MSN'], nPops['MSN'], con_spec, syn_spec)
    if args.Nf > 0:
        con_spec = {'rule': 'fixed_indegree', 'indegree': 100}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[-1][0], 'delay': 1.0}
        nest.Connect(nPops['FSI'], nPops['MSN'], con_spec, syn_spec)

    if W > 0:
        # cortical stimulus (excitatory)
        con_spec = {'rule': 'fixed_indegree', 'indegree': 20}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][0], 'delay': 2.0}
        nest.Connect(stm, nPops['MSN'], con_spec, syn_spec)
        if args.Nf > 0:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][-1], 'delay': 1.0}
            nest.Connect(stm, nPops['FSI'], con_spec, syn_spec)

    # record device
    print('link devices')
    # 'record_to': 'ascii'
    spikeDetector = nest.Create('spike_recorder', 1, {'label':recpath + 'spk'})
    for npop in nPops.values():
        nest.Connect(npop, spikeDetector)
    
    # subthreshold recording
    if args.Nf == 0:
        del nTypes[-1]
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

    # visualize
    print('visualize')
    # spikes
    spikeEvents = nest.GetStatus(spikeDetector, 'events')[0]
    spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']
    spikeIds -= stm[-1]
    mask = (spikeIds > nNums[0]-100) & (spikeIds <= nNums[0] + 100)
    visSpk(spikeTimes[mask], spikeIds[mask]-nNums[0]+100, path=spkpath + 'spk')

    # firing rates
    if nNums[-1] == 0:
        nNums[-1] = 1
    idBins = np.concatenate([[0], np.cumsum(nNums)]) + 1
    tmBins = np.arange(0, simtime*args.epoch+1, 100)
    rateTimePop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0]*10/nNums
    visCurve([tmBins[:-1]]*len(nTypes), rateTimePop.T,
             'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_', nTypes)
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