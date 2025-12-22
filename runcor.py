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
from lib.mcalc import gcf2ann

# simulation libs
from lib import params
import nest

def main():   
    """Interneuron motif PC-PV-SOM-VIP
    """
    # create argument parser
    parser = argparse.ArgumentParser(description='Striatal Microcircuit.')
    # profiling
    parser.add_argument('--dpath', type=str, help='data path', default='./data')
    parser.add_argument('--fpath', type=str, help='data path', default='./plot')
    # simulation setting
    parser.add_argument('--epoch', type=int, help='number of epochs', default=2)
    parser.add_argument('--T', type=int, help='simulation time', default=1000)
    parser.add_argument('--Nm', type=int, help='number of MSNs', default=2500)
    parser.add_argument('--Nf', type=int, help='number of FSIs', default=0)
    # input setting
    parser.add_argument('--W', type=float, help='within-pool correlation', default=0.1)
    parser.add_argument('--B', type=float, help='between-pool correlation', default=0.1)
    parser.add_argument('--D', type=float, help='inhibitory synapse delay', default=2.0)
    parser.add_argument('--delay-width', type=float, help='half-width for uniform delay jitter (ms)', default=1.0)
    parser.add_argument('--weight-sigma', type=float, help='sigma for lognormal weight jitter', default=0.5)
    parser.add_argument('--nthreads', type=int, help='number of local NEST threads', default=1)
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
    mode = 'rec_0_str'
    suffixpath = '/%s/Nf%d-W%s-B%s/'%(mode, args.Nf, args.W, args.B)
    recpath = args.dpath + suffixpath                                   # path for saving recordings (gdf files)
    os.makedirs(recpath, exist_ok=True)
    if args.viz:
        figpath = args.fpath + suffixpath                                   # path for saving figures
        spkpath = figpath + 'spk/'
        ratpath = figpath + 'rat/'
        wtspath = figpath + 'wt/'
        for path in [spkpath, ratpath, wtspath]:
            os.makedirs(path, exist_ok=True)

    print('Simulation start ... ... ')
    print(suffixpath)

    #* Motif network
    # neurons
    nTypes = ['MSN', 'FSI']
    nNums = [args.Nm, args.Nf]
    paramMSN = params.paramMSN
    paramFSI = params.paramFSI

    MSN_deg = int(nNums[0]/10)
    FSI_deg = int(nNums[0]/100*0.6)
    CTX_deg = 100

    allow_autapses = True
    allow_multapses = True

    # input
    print('inputs')
    B, W = args.B, args.W

    # generate input spike trains
    msd = 4294967295
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)

    if W > 0:
        nstm = int(CTX_deg / W)
        # partially overlapped cortical sources
        stm = nest.Create('poisson_generator', int(nstm*(2-B)), params={'rate': 10.0})
        # input recording
        inDetector = nest.Create('spike_recorder', 1)
        nest.Connect(stm, inDetector)
        # run simulation
        nest.Simulate(simtime)
        # get spike trains
        inSpks = nest.GetStatus(inDetector, 'events')[0]

        # # visualize input spikes
        # plt.figure(figsize=(6,4))
        # visSpk(inSpks['times'], inSpks['senders'], path='./input')
        # assert False

        print(np.unique(inSpks['senders']).size)
        print(inSpks)

    Js = [  [2.0, 1.0],
            [4.8, 0.5],
            [0.03, 0.],
            [0.5, 0.]]

    # reset kernel
    msd = int(time.time() * 1000.0) % 4294967295
    nest.ResetKernel()
    nest.set_verbosity("M_WARNING")
    nest.set(resolution=0.1, rng_seed=msd)
    # set number of local threads to speed up NEST operations
    try:
        nest.SetKernelStatus({'local_num_threads': max(1, int(args.nthreads))})
    except Exception:
        pass

    # neurons
    print('create neurons')
    # scale = 1/30
    # scale = 1/60
    scale = 0
    # heterogeneous parameters
    param_keys = ['C_m', 'tau_syn_in', 't_ref', 'V_th', 'g_L']

    nPops = {}
    nPops['MSN'] = nest.Create('iaf_cond_alpha', nNums[0])
    if scale > 0:
        sample_msn = params.sample_params_normal(paramMSN, designated_keys=param_keys, scale=scale, n_samples=nNums[0])
        nest.SetStatus(nPops['MSN'], sample_msn)
    else:
        nest.SetStatus(nPops['MSN'], paramMSN)

    if nNums[-1] > 0:
        nPops['FSI'] = nest.Create('iaf_cond_alpha', nNums[-1])
        if scale > 0:
            sample_fsi = params.sample_params_normal(paramFSI, designated_keys=param_keys, scale=scale, n_samples=nNums[-1])
            nest.SetStatus(nPops['FSI'], sample_fsi)
        else:
            nest.SetStatus(nPops['FSI'], paramFSI)

    # background noise
    bkgF = nest.Create('poisson_generator', 1, {'rate': 5.75e3})
    if args.Nf == 0 and args.W > 0:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 3.2e3})
    else:
        bkgM = nest.Create('poisson_generator', 1, {'rate': 5.95e3})
    # cortical input
    if W > 0:
        stm = nest.Create('spike_generator', int(nstm*(2-B)))
        for i in range(int(nstm*(2-B))):
            spks = inSpks['times'][inSpks['senders'] == (i + 1)]
            stm[i].spike_times = np.concatenate([spks + e*simtime for e in range(args.epoch)])

    # connections
    print('generating connections')
    # background input (excitatory)
    nest.Connect(bkgM, nPops['MSN'], syn_spec={'weight': Js[0][0], 'delay': 1.0})
    if args.Nf > 0:
        nest.Connect(bkgF, nPops['FSI'], syn_spec={'weight': Js[0][-1], 'delay': 1.0})
    if W > 0:
        # cortical stimulus (excitatory)
        con_spec = {'rule': 'fixed_indegree', 'indegree': CTX_deg, 'allow_multapses': allow_multapses}
        syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][0], 'delay': 1.0}
        nest.Connect(stm[:nstm], nPops['MSN'][:nNums[0]//2], con_spec, syn_spec)
        nest.Connect(stm[-nstm:], nPops['MSN'][nNums[0]//2:], con_spec, syn_spec)
        if args.Nf > 0:
            syn_spec ={'synapse_model': 'static_synapse', 'weight': Js[1][-1]/2, 'delay': 1.0}
            nest.Connect(stm[:nstm], nPops['FSI'], con_spec, syn_spec)
            nest.Connect(stm[-nstm:], nPops['FSI'], con_spec, syn_spec)

    # Recurrent inhibitory MSN→MSN connections
    con_spec = {
        'rule': 'fixed_indegree',
        'indegree': MSN_deg,
        'allow_autapses': False,
        'allow_multapses': allow_multapses
    }
    # Compute lognormal mu
    mu_msn = np.log(abs(Js[2][0])) - 0.5 * args.weight_sigma**2
    syn_spec = {
        "synapse_model": "static_synapse",
        "weight": -1.0 * nest.random.lognormal(mean=mu_msn, std=args.weight_sigma),
        "delay": nest.random.uniform(
            min=max(0.1, args.D - args.delay_width),
            max=args.D + args.delay_width
        ),
    }
    nest.Connect(nPops["MSN"], nPops["MSN"], con_spec, syn_spec)
        
    # Feedforward inhibitory FSI→MSN connections
    if args.Nf > 0:
        con_spec = {'rule': 'fixed_indegree', 'indegree': FSI_deg, 'allow_multapses': allow_multapses, 'allow_autapses': allow_autapses}
        # Compute lognormal mu
        mu_fsi = np.log(abs(Js[-1][0])) - 0.5 * args.weight_sigma**2
        syn_spec = {
            "synapse_model": "static_synapse",
            "weight": -1.0 * nest.random.lognormal(mean=mu_fsi, std=args.weight_sigma),
            "delay": nest.random.uniform(
                min=max(0.1, args.D - args.delay_width),
                max=args.D + args.delay_width
            ),
        }
        nest.Connect(nPops["FSI"], nPops["MSN"], con_spec, syn_spec)

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
    
    # Select 10 neurons per type
    subNeurons = [nPops[ntype][:25] for ntype in nTypes]
    mulDetects = [
        nest.Create('multimeter', params={'record_from': ['V_m', 'g_ex', 'g_in'], 'interval': 0.1}, n=len(subNeurons[i]))
        for i, ntype in enumerate(nTypes)
    ]
    for neurons, detects in zip(subNeurons, mulDetects):
        for det, nrn in zip(detects, neurons):
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
    print('number of spikes: ', len(spikeEvents['times']))
    spikeTimes, spikeIds = spikeEvents['times'], spikeEvents['senders']

    # visualize
    print('visualize')
    if args.viz:
        # # input spikes
        # if W > 0:
        #     pulseEvents = nest.GetStatus(inDetector, 'events')[0]
        #     pulseTimes, pulseIds = pulseEvents['times'], pulseEvents['senders']
        #     visSpk(pulseTimes, pulseIds-nNums[0], path=spkpath + 'input')

        mask = (spikeIds > nNums[0]-100) & (spikeIds <= nNums[0] + 100)
        visSpk(spikeTimes[mask], spikeIds[mask]-nNums[0]+100, path=spkpath + 'spk')

        # membrane potentials
        multiEvents = []
        for detects in mulDetects:  # mulDetects is a list of lists
            events_per_type = [nest.GetStatus(det)[0]['events'] for det in detects]
            multiEvents.append(events_per_type)
        multiPlot(multiEvents, [0., simtime], nTypes, ratpath + 'vg_')

        # # firing rates
        # nNums = [nNums[0]//2, nNums[0]//2, args.Nf]
        # nTypes = ['M1', 'M2', 'FSI']
        # if args.Nf == 0:
        #     del nNums[-1]
        #     del nTypes[-1]
        # binsize = 100
        # tmBins = np.arange(0, simtime*args.epoch+1, binsize)
        # idBins = np.concatenate([[0], np.cumsum(nNums)]) + 1
        # ratesPop = np.histogram2d(spikeTimes, spikeIds, bins=[tmBins, idBins])[0] * 1e3 / binsize / nNums
        # visCurve([tmBins[:-1]]*len(nTypes), ratesPop.T,
        #         'time (ms)', 'rate (Hz)', 'firing rate', ratpath + 'fr_', nTypes)
        # print('mean firing rate: ', np.mean(ratesPop, axis=0))

    # save data
    np.savez(recpath + 'spk.npz', spikeIds, spikeTimes)

    # Save multimeter recordings
    data_dict = {}
    for type_idx, detects in enumerate(mulDetects):
        for det_idx, det in enumerate(detects):
            events = nest.GetStatus(det)[0]['events']
            key_prefix = f"{nTypes[type_idx]}_neuron{det_idx}"
            for key, values in events.items():
                data_dict[f"{key_prefix}_{key}"] = values
    filename = recpath + "mem.npz"
    np.savez(filename, **data_dict)

    print('done')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))