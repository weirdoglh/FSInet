""" A cortical neuron motif simulator
"""

# system
import time
from datetime import datetime
from tqdm import tqdm

# tools
import matplotlib.pyplot as plt

# computation libs
import numpy as np


# simulation libs
from lib import params
import nest

def main():
    nparam = params.paramSin
    nparam['E_L'] = -57.0

    # initialization of NEST
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.SetKernelStatus({'overwrite_files':True, 'local_num_threads':1})

    # neuron
    print('create neurons')
    nrn = nest.Create('iaf_cond_alpha', 1, nparam)

    # connections
    print('generating connections')
    wscales = np.linspace(0., 1.0, 101)
    ntrial = len(wscales)
    simtime = 50.
    sg = nest.Create('spike_generator', 1)
    nest.Connect(sg, nrn)

    # record device
    print('link devices')
    multiDetector = nest.Create('multimeter', 1, {'interval': 0.1, 'record_from': ['V_m']})
    nest.Connect(multiDetector, nrn)

    # simulations
    print('init states')
    offset = 10.0
    for tr in tqdm(range(ntrial)):
        nest.SetStatus(nrn, params='V_m', val=nparam['E_L'])

        conn = nest.GetConnections(sg, nrn)
        nest.SetStatus(conn, {'weight': wscales[tr]})
        nest.SetStatus(sg, {'spike_times':np.array([tr*simtime + offset])})

        nest.Simulate(simtime)

    offset = 10.0 + ntrial*simtime
    for tr in tqdm(range(ntrial)):
        nest.SetStatus(nrn, params='V_m', val=nparam['E_L'])

        conn = nest.GetConnections(sg, nrn)
        nest.SetStatus(conn, {'weight': -wscales[tr]})
        nest.SetStatus(sg, {'spike_times':np.array([tr*simtime + offset])})
        
        nest.Simulate(simtime)

    multiEvents = nest.GetStatus(multiDetector)[0]['events']
    ts = multiEvents['times']
    vm = multiEvents['V_m']
    synvol = np.zeros((2, ntrial))
    offset = 10.0
    for tr in range(ntrial):
        synvol[0, tr]  = np.max(vm[(ts>tr*simtime + offset) & ((ts<(tr+1)*simtime + offset))]) - nparam['E_L']
    offset = 10.0 + ntrial*simtime
    for tr in range(ntrial):
        synvol[1, tr]  = np.min(vm[(ts>tr*simtime + offset) & ((ts<(tr+1)*simtime + offset))]) - nparam['E_L'] 

    print(synvol)

    numtype = 1
    plt.subplots(1,numtype, figsize=(5, 2.5))
    for fidx in range(numtype):
        plt.subplot(1,numtype, fidx+1)
        plt.plot(wscales, synvol[0], c='b', label='EPSP')
        plt.plot(wscales, synvol[1], c='r', label='IPSP')
        if fidx == 0:
            plt.xlabel(r'$g_{syn}$', fontsize='large', labelpad=-10)
            plt.ylabel('PSP (mV)', fontsize='large')
            plt.yticks([-1.0, -0.5, 0., 0.2])
            plt.xticks([0., 1.0])
        else:
            plt.xticks([])
            plt.yticks([])
        if fidx == 1:
            plt.legend()
        plt.ylim([-1, 0.2])
    plt.tight_layout()
    plt.savefig('./plot/synvol.pdf', dpi=1000)

    plt.subplots(numtype,1, figsize=(8.0, 10.0))
    for fidx in range(numtype):
        plt.subplot(numtype,1,fidx+1)
        plt.plot(ts[ts<simtime*ntrial], vm[ts<simtime*ntrial], c='b', label='EPSP')
        plt.plot(ts[ts>simtime*ntrial]-simtime*ntrial, vm[ts>simtime*ntrial], c='r', label='IPSP')
        plt.xticks([0, simtime*ntrial/4, simtime*ntrial*3/4, simtime*ntrial], ['0', '0.25', '0.75', '1.0'])
        plt.xlabel(r'$g_{syn}$', labelpad=-10)
        plt.ylabel('Vm (mV)')
        plt.legend()
    plt.tight_layout()
    plt.savefig('./plot/synvm.pdf', dpi=100)

    factors = np.zeros(2)
    for ii in range(2):
        factors[ii] = 1/np.abs((synvol[ii,0]-synvol[ii,-1])/(wscales[0]-wscales[-1]))
    print(1/factors)
    
    np.savez('./lib/synvol.npz', ws=wscales, synvol=synvol, factors=factors)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))