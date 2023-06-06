""" A cortical neuron motif simulator
"""

# system
import time
from datetime import datetime
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
    parser.add_argument('--test', action='store_true', default=False)
    # simulation setting
    parser.add_argument('--epoch', type=int, help='number of epochs', default=2)
    parser.add_argument('--T', type=int, help='simulation time', default=3000)
    parser.add_argument('--Nm', type=int, help='number of MSNs', default=5000)
    parser.add_argument('--Nf', type=int, help='number of FSIs', default=100)
    # input setting
    parser.add_argument('--W', type=float, help='within-pool correlation', default=0.1)
    parser.add_argument('--B', type=float, help='between-pool correlation', default=0.1)
    parser.add_argument('--D', type=float, help='recurrent connection delay', default=1.0)
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
    if args.test:
        suffixpath = '/baseline/'
    else:
        suffixpath = '/cor-D%.1f/Nf%d-W%.2f-B%.2f/'%(args.D, args.Nf, args.W, args.B)
    recpath = args.dpath + suffixpath                                   # path for saving recordings (gdf files)
    figpath = args.fpath + suffixpath                                   # path for saving figures
    spkpath = figpath + 'spk/'
    ratpath = figpath + 'rat/'
    wtspath = figpath + 'wt/'
    for path in [recpath, spkpath, ratpath, wtspath]:
        os.makedirs(path, exist_ok=True)

    #* Motif network
    # neurons
    nTypes = ['M1', 'M2', 'FSI']
    nNums = [args.Nm, args.Nm, args.Nf]
    nparam = {'V_th': -55., 'V_reset': -70.}
    nParams = [nparam, nparam, nparam]

    # connection strengths
    Js = [  [1.1, 1.1, 1.1],
            [2.0, 2.0, 0.8],
            [0.3, 0.3, 0.],
            [0.3, 0.3, 0.],
            [0.2, 0.2, 0.]]

    # initialization of NEST
    msd = int(datetime.now().strftime("%m%d%H%M%S"))
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    # neuron
    print('create neurons')
    nPops = {}
    for ntype, nnum, nparam in zip(nTypes, nNums, nParams):
        if (ntype == 'FSI') & (nnum == 0):
            break
        else:
            nPops[ntype] = nest.Create('iaf_cond_alpha', nnum)
            nest.SetStatus(nPops[ntype], nparam)

    # input
    print('inputs')
    # background
    if args.Nf > 0:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 6.9e3})
    else:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 6.2e3})
    bkgF = nest.Create('poisson_generator', 1, {'rate': 6.9e3})
    # stimulus
    B, W = args.B, args.W
    assert B > 0
    mip = nest.Create('mip_generator', params={'rate': 40./B/W, 'p_copy': B})
    # pool holder
    pools = nest.Create('parrot_neuron', 2)
    nest.Connect(mip, pools)
    # dilutor
    dilutors = nest.Create('spike_dilutor', 2, params={'p_copy': W})
    nest.Connect(pools, dilutors, 'one_to_one')
    # spike trains
    stm1 = nest.Create('parrot_neuron', 1000)
    stm2 = nest.Create('parrot_neuron', 1000)
    nest.Connect(dilutors[0], stm1)
    nest.Connect(dilutors[1], stm2)  

    # connections
    print('generating connections')
    # background input (excitatory)
    nest.Connect(bkgM, nPops['M1'] + nPops['M2'], syn_spec={'weight': Js[0][0], 'delay': 1.0})
    if args.Nf > 0:
        nest.Connect(bkgF, nPops['FSI'], syn_spec={'weight': Js[0][-1], 'delay': 1.0})

    # striatal connection (inhibitory)
    con_spec = {'rule': 'fixed_indegree', 'indegree': 50}
    syn_spec = {'synapse_model': 'static_synapse', 'weight': -Js[2][0], 'delay': args.D}
    nest.Connect(nPops['M1']+nPops['M2'], nPops['M1']+nPops['M2'], con_spec, syn_spec)
    if args.Nf > 0:
        con_spec = {'rule': 'fixed_indegree', 'indegree': 100}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': -Js[-1][0], 'delay': 1.0}
        nest.Connect(nPops['FSI'], nPops['M1'] + nPops['M2'], con_spec, syn_spec)

    if not args.test:
        # cortical stimulus (excitatory)
        con_spec = {'rule': 'fixed_indegree', 'indegree': 20}
        if args.Nf > 0:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][0], 'delay': 1.0}
        else:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': 0.8*Js[1][0], 'delay': 1.0}
        nest.Connect(stm1, nPops['M1'], con_spec, syn_spec)
        nest.Connect(stm2, nPops['M2'], con_spec, syn_spec)
        if args.Nf > 0:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][-1], 'delay': 1.0}
            nest.Connect(stm1 + stm2, nPops['FSI'], con_spec, syn_spec)

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
    for e in range(args.epoch):
        # initial state
        for npop in nPops.values():
            rvs = np.random.uniform(-70., -55., len(npop))
            nest.SetStatus(npop, params='V_m', val=rvs.tolist())

        # simulate
        nest.Simulate(simtime)

    # visualize
    print('visualize')
    # spikes
    spikeEvents = nest.GetStatus(spikeDetector, 'events')[0]
    spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']
    mask = (spikeIds > np.sum(nNums)-200) & (spikeIds <= np.sum(nNums))
    visSpk(spikeTimes[mask], spikeIds[mask]-np.sum(nNums)+200, path=spkpath + 'spk')

    # firing rates
    idBins = np.concatenate([[0], np.cumsum(nNums)]) + 1
    tmBins = np.arange(0, simtime*args.epoch+1, 500)
    rateTimePop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0]*2/nNums
    visCurve([tmBins[:-1]]*len(nTypes), rateTimePop.T,
             'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_', nTypes)

    # multimeter
    multiEvents = [nest.GetStatus(detector)[0]['events'] for detector in mulDetects]
    multiPlot(multiEvents, [0., simtime], nTypes, ratpath + 'vg_')

    np.savez(recpath + 'spk.npz', spikeIds, spikeTimes)
    print('done')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))