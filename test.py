
'''
    coments
    '''
from sre_constants import NOT_LITERAL_IGNORE
import nest
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

delay = 1.
J = 0.1   # postsynaptic amplitude in mV
#################################################################################
# SYNAPTIC WEIGHTS
J_exd1 = 0.15       # amplitude of excitatory postsynaptic current in D1
J_exd2 = 0.1       # amplitude of excitatory postsynaptic current in D2
J_d1d1 = 0.07       # amplitude of excitatory postsynaptic current in D1-->D1
J_d1d2 = 0.15       # amplitude of excitatory postsynaptic current in D2-->D1
J_d2d1 = 0.07       # amplitude of excitatory postsynaptic current in D1-->D2
J_d2d2 = 0.15       # amplitude of excitatory postsynaptic current in D1-->D2

## Enter here the synaptic weights
J_ex = 1.1
J_in = 0.6

no_trial = 2

ex_msn_rate  = 7.
ex_fsi_rate  = 7.2

Nmsn = 400

spk_group_1  = 1000
spk_group_2  = 1000
Nctx = spk_group_1 + spk_group_2

Kc2m = 20
Kf2m = 20
Km2m = 50

warm_time = 500.
sim_time = 2e3

ctx_corr = 0.2
ctx_rate = 80.
dt = 0.1 #ms


Nfs = [0, 20, 40, 60, 80, 100]
W = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
B = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
input_rate = [30., 40., 50.]


W = [0.1]
B = [0.1]
Nfs = [20]
j_fb = [0.5]
input_rate = [40.]

# test = True
test = False

for yy in range(len(j_fb)):
    for ww in range(len(input_rate)):
        for ii in range(len(Nfs)):
            for jj in range(len(B)):
                for kk in range(len(W)):
                    ctx_rate = input_rate[ww]
                    if Nfs[ii]==0:
                        ex_msn_rate  = 6.2
                        ex_fsi_rate  = 7.2
                    else:
                        ex_msn_rate  = 7.
                        ex_fsi_rate  = 7.2
                    
                    fname = 'FCount_' + str(Nfs[ii]) + '_CtxRate_' + str(input_rate[ww]) + '_B_' + str(B[jj]) + '_W_' + str(W[kk]) + '_JFB_' + str(j_fb[yy])
                    print(fname)
                    nest.ResetKernel()
                    nest.SetKernelStatus({
                                        'local_num_threads': 1,
                                        'resolution': 0.1,
                                        'data_path': './data/test/',
                                        'overwrite_files': True
                                        })

                    sd = nest.Create('spike_detector',1)
                    nest.SetStatus(sd,{'label':fname, 'to_file': True, 'to_memory': False})
                    m = nest.Create('multimeter', params = {'withtime': True, 'interval': 0.1, 'record_from': ['V_m','g_ex','g_in']})
                    nest.SetStatus(m,{'label':fname, 'to_file': True, 'to_memory': False,'start':400.,'stop':3000.})

                    # Store the correlated spikes in spike generators
                    ctx_spk_gen = nest.Create('spike_generator', spk_group_1+spk_group_2)

                    syn_params_ex = {"weight": J_ex, "delay": delay}
                    syn_params_ex_fs = {"weight": J_ex, "delay": delay}
                    syn_params_in = {"weight": -J_in*1.5, "delay": delay}
                    syn_params_msn_msn = {"weight": -J_in*j_fb[yy], "delay": delay}

                    if Nfs[ii]==0:
                        syn_params_ex_msn = {"weight": 0.8*J_ex, "delay": delay+1.}
                    else:
                        syn_params_ex_msn = {"weight": J_ex, "delay": delay+1.}
                    

                    # create neurons etc.
                    msn = nest.Create("iaf_cond_alpha", Nmsn)        
                    nest.SetStatus(msn,{'V_th':-55.,'V_reset':-70.})
                    print(nest.GetStatus([msn[0]]))

                    if Nfs[ii]!=0:
                        fsi = nest.Create("iaf_cond_alpha", Nfs[ii])
                        nest.SetStatus(fsi,{'V_th':-55.,'V_reset':-70.})

                    conn_dict = {"rule": "fixed_indegree", "indegree": Kc2m}
                    if not test:
                        nest.Connect(ctx_spk_gen[0:spk_group_1],msn[:200],conn_dict,syn_params_ex_msn)
                        nest.Connect(ctx_spk_gen[spk_group_1:],msn[200:],conn_dict,syn_params_ex_msn)
                    
                    # MSN-MSN connectivity
                    conn_dict_msn2msn = {"rule": "fixed_indegree", "indegree": Km2m}
                    nest.Connect(msn,msn,conn_dict_msn2msn,syn_params_msn_msn)

                    # FSI-MSN connectivity
                    if Nfs[ii]!=0:
                        # cortext to FSI
                        if not test:
                            nest.Connect(ctx_spk_gen,fsi,conn_dict,syn_params_ex_fs)
                        # FSI to MSN
                        conn_dict_fsi2msn = {"rule": "fixed_indegree", "indegree": Kf2m}
                        nest.Connect(fsi,msn,conn_dict_fsi2msn,syn_params_in)

                    # Also inject some Poisson input to the MSN and FSIs
                    rate_exc_msn = 1e3*ex_msn_rate 
                    rate_exc_fsi = 1e3*ex_fsi_rate 

                    poi_ex_msn = nest.Create('poisson_generator',1,{'rate':rate_exc_msn})
                    poi_ex_fsi = nest.Create('poisson_generator',1,{'rate':rate_exc_fsi})

                    nest.Connect(poi_ex_msn, msn, {'rule': 'all_to_all'}, syn_params_ex)
                    if Nfs[ii]!=0:
                        nest.Connect(poi_ex_fsi, fsi, {'rule': 'all_to_all'}, syn_params_ex)

                    # subthrehsold
                    nest.Connect(m, msn[0:10])
                    if Nfs[ii]!=0:
                        nest.Connect(m, fsi[0:5])

                    # spiking
                    nest.Connect(msn,sd)

                    if Nfs[ii]!=0:
                        nest.Connect(fsi,sd)

                    # # Prepare MIPs for the two mothers
                    mother_rate = ctx_rate / W[kk]
                    grandmother_rate = mother_rate / B[jj]

                    for tt in range(no_trial):
                        print('trial = ', str(tt+1),'of',str(no_trial),'trials')
                        warmup = warm_time + tt*(sim_time+2*warm_time)

                        gm_rate = grandmother_rate*dt*1e-3
                        gm_spk_count = np.random.poisson(gm_rate*sim_time/dt)
                        gm_spk = np.random.uniform(0,1,gm_spk_count)*sim_time
                        gm_spk = np.sort(gm_spk)
                        # first mother
                        no_spk = np.random.poisson(mother_rate*sim_time/1e3)
                        rand_id = np.random.permutation(len(gm_spk))
                        spk_time = gm_spk[rand_id[0:no_spk]]
                        spk_time = np.round(spk_time*10)/10
                        mo_spk_1 = np.sort(spk_time)
                        # second mother
                        #no_spk = np.random.poisson(mother_rate*sim_time/1e3)
                        rand_id = np.random.permutation(len(gm_spk))
                        spk_time = gm_spk[rand_id[0:no_spk]]
                        spk_time = np.round(spk_time*10)/10
                        mo_spk_2 = np.sort(spk_time)

                        mip_spk_1 = []
                        for nn in range(spk_group_1):
                            no_spk = np.random.poisson(ctx_rate*sim_time/1e3)
                            rand_id = np.random.permutation(len(mo_spk_1))
                            spk_time = mo_spk_1[rand_id[0:no_spk]]
                            spk_time = np.round(spk_time*10 + 1.*np.random.normal(0.,2.,len(spk_time)))/10
                            spk_time=np.sort(spk_time)
                            mip_spk_1.append(spk_time)
                            nest.SetStatus([ctx_spk_gen[nn]],{'spike_times': spk_time+warmup})
                        
                        mip_spk_2 = []
                        for nn in range(spk_group_2):
                            no_spk = np.random.poisson(ctx_rate*sim_time/1e3)
                            rand_id = np.random.permutation(len(mo_spk_2))
                            spk_time = mo_spk_2[rand_id[0:no_spk]]
                            spk_time = np.round(spk_time*10 + 1.*np.random.normal(0.,2.,len(spk_time)))/10
                            spk_time=np.sort(spk_time)
                            mip_spk_2.append(spk_time)
                            nest.SetStatus([ctx_spk_gen[nn+spk_group_1]],{'spike_times': spk_time+warmup})

                        nest.Simulate(sim_time + 2*warm_time)
