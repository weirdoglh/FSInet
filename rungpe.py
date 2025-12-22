import numpy as np
import os
import pandas as pd
import  matplotlib.pyplot as plt
import nest
from collections import defaultdict

import argparse

from lib import params

# create argument parser
parser = argparse.ArgumentParser(description='Striatal Microcircuit.')
parser.add_argument('--Nf', type=int, help='number of FSIs', default=0)
parser.add_argument('--W', type=float, help='within-pool correlation', default=0.1)
parser.add_argument('--B', type=float, help='between-pool correlation', default=0.1)
parser.add_argument('--rate', type=float, help='base firing rate', default=7e3)
parser.add_argument('--epoch', type=int, help='number of epochs', default=100)
parser.add_argument('--viz', action='store_true', help='visualize connectivity and spike raster')
args = parser.parse_args()

# data setting
Nm = 1250
nMSN = Nm // 10

epoch = args.epoch
T, dt = 2500, 0.1

Nf = args.Nf
W = args.W
B = args.B
base_input_rate = args.rate

recpar = 'rec_0_str'
recpath = './data/%s/msn'%recpar
savepath = './data/%s/gpe'%recpar
label = '/Nf%d-W%s-B%s/'%(Nf, W, B)
os.makedirs(savepath + label , exist_ok=True)

def run_downstream_sim(es, ts, Nm=1250, T=2500.0, syn_weight=-0.1, noise_rate=7e3, display_plots=True):
    """Run a NEST simulation that drives one postsynaptic neuron from spike trains (es, ts).
    Returns a dict with summary statistics and recorded traces.
    """
    nest.ResetKernel()
    spike_times = ts
    spike_ids = es.astype(int)
    by_sender = defaultdict(list)
    for sid, t in zip(spike_ids, spike_times):
        by_sender[sid].append(float(t))
    senders = sorted(by_sender.keys())
    spike_time_lists = [by_sender[s] for s in senders]

    # create generators and set spike times
    generators = nest.Create("spike_generator", len(senders))
    nest.SetStatus(generators, [{'spike_times': st} for st in spike_time_lists])
    spike_recorder = nest.Create("spike_recorder")
    nest.Connect(generators, spike_recorder)

    # gpe neurons
    neuron = nest.Create("iaf_cond_alpha")
    nest.SetStatus(neuron, params.paramGPE)

    bkg = nest.Create('poisson_generator', 1, {'rate': noise_rate})
    nest.Connect(bkg, neuron, syn_spec={'weight': 1.0, 'delay': 1.0})

    # sample weight from log-normal distribution
    std = 0.5
    mu_gpe = np.log(abs(syn_weight)) - 0.5 * std**2
    syn_spec = {
        "synapse_model": "static_synapse",
        "weight": -1.0 * nest.random.lognormal(mean=mu_gpe, std=std),
        "delay": nest.random.uniform(min=1.0, max=3.0),
    }
    nest.Connect(generators[:Nm], neuron, syn_spec=syn_spec)

    voltmeter = nest.Create("voltmeter")
    nest.Connect(voltmeter, neuron)
    neuron_spike_recorder = nest.Create("spike_recorder")
    nest.Connect(neuron, neuron_spike_recorder)

    # Suppress NEST output
    nest.set_verbosity("M_WARNING")
    nest.Simulate(T)

    # collect statistics
    vm_events = nest.GetStatus(voltmeter, 'events')[0] if voltmeter else {'V_m': np.array([]), 'times': np.array([])}
    vm_values = vm_events.get('V_m', np.array([]))
    vm_times = vm_events.get('times', np.array([]))
    vm_mean = float(np.mean(vm_values)) if vm_values.size else float('nan')
    vm_var = float(np.var(vm_values)) if vm_values.size else float('nan')
    neuron_events = nest.GetStatus(neuron_spike_recorder, 'events')[0]
    neuron_spike_times = neuron_events.get('times', [])
    n_spikes = len(neuron_spike_times)

    # optional plotting
    if display_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(vm_times, vm_values, c='orange')
        plt.xlim(0, T)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane potential (mV)")
        plt.title("Membrane Potential of Postsynaptic Neuron")
        plt.savefig(savepath + label + 'vm_plot.png', dpi=300)
        plt.close()
    return { 'vm_mean': vm_mean, 'vm_var': vm_var, 'n_spikes': n_spikes, 'spike_times': neuron_spike_times }

data = np.load(recpath + label + 'spk.npz')
es, ts = data.f.arr_0, data.f.arr_1

sim_stats = run_downstream_sim(es, ts, Nm=Nm, T=T*epoch, syn_weight=-0.02, noise_rate=base_input_rate, display_plots=args.viz)

# save output spikes
np.savez(savepath + label + 'spk.npz', sim_stats['spike_times'])

# extract stats
print("Downstream neuron stats for Nf=%d, W=%.2f, B=%.2f:" % (Nf, W, B))
print("  Mean membrane potential: %.2f mV" % sim_stats['vm_mean'])
print("  Variance of membrane potential: %.2f mV^2" % sim_stats['vm_var'])
print("  Firing rate: %.2f" % (sim_stats['n_spikes']/(epoch*T/1000.0)))  # spikes per second