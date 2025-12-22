# Parameter setting of neurons
paramSin = {'C_m': 250., 'E_L': -70., 'g_L': 50/3, 'V_th': -55., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in':15.0}

# paramMSN = {'C_m': 80., 'E_L': -80., 'g_L': 10.0, 'V_th': -45., 't_ref': 2., 'V_reset': -80., 'tau_syn_ex': 0.2, 'tau_syn_in': 15.0}
# paramFSI = {'C_m': 70., 'E_L': -70., 'g_L': 5.0, 'V_th': -40., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in': 15.0}

paramMSN = {'C_m': 80., 'E_L': -80., 'g_L': 10.0, 'V_th': -45., 't_ref': 2., 'V_reset': -70., 'tau_syn_ex': 0.2, 'tau_syn_in': 15.0}
paramFSI = {'C_m': 70., 'E_L': -70., 'g_L': 5.0, 'V_th': -40., 't_ref': 2., 'V_reset': -60., 'tau_syn_ex': 0.2, 'tau_syn_in': 15.0}
paramGPE = {'C_m': 70., 'E_L': -70., 'g_L': 2.5, 'V_th': -45., 't_ref': 2., 'V_reset': -60., 'tau_syn_ex': 0.2, 'tau_syn_in': 15.0}

import numpy as np

# Utility: sample numeric parameters uniformly around nominal values
# Only perturb designated keys; keep all other parameters equal to nparam
# Returns a single dict when n_samples==1 (default), otherwise a list of dicts.
def sample_params_uniform(base_params, designated_keys=None, scale=0.1, n_samples=1):
    if designated_keys is None:
        designated_keys = ['C_m', 'g_L', 'V_th', 't_ref', 'tau_syn_in']

    def _single_sample():
        samp = dict(base_params)
        for k in designated_keys:
            if k in base_params and isinstance(base_params[k], (int, float, np.integer, np.floating)):
                v = base_params[k]
                low = v * (1 - scale)
                high = v * (1 + scale)
                val = np.random.uniform(low, high)
                if isinstance(v, int):
                    val = int(round(val))
                samp[k] = val
        return samp

    if n_samples is None or int(n_samples) <= 1:
        return _single_sample()
    else:
        samples = []
        for _ in range(int(n_samples)):
            samples.append(_single_sample())
        return samples
    

# Utility: sample numeric parameters from a normal distribution around nominal values
# Only perturb designated keys; keep all other parameters equal to nparam
def sample_params_normal(base_params, designated_keys=None, scale=0.1, n_samples=1):
    if designated_keys is None:
        designated_keys = ['C_m', 'g_L', 'V_th', 't_ref', 'tau_syn_in']

    def _single_normal():
        samp = dict(base_params)
        for k in designated_keys:
            if k in base_params and isinstance(base_params[k], (int, float, np.integer, np.floating)):
                v = base_params[k]
                sd = abs(v) * scale if v != 0 else scale
                val = np.random.normal(loc=v, scale=sd)
                if isinstance(v, int):
                    val = int(round(val))
                samp[k] = val
        return samp

    if n_samples is None or int(n_samples) <= 1:
        return _single_normal()
    else:
        samples = []
        for _ in range(int(n_samples)):
            samples.append(_single_normal())
        return samples