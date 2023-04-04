# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 00:02:42 2022

@author: Umt
"""

import numpy as np

class Fronthaul():
    def __init__(self, state, channel_fun, cpu_ant, cpu_phases, num_groups=1):
        self.cpu_phases = cpu_phases
        self.L = cpu_ant
        self.H = self.generate_fronthaul_channel(state, channel_fun) # M x L
        self.M = self.H.shape[0]
        self.K = num_groups
    
    # Optimize the rates of the groups with iterative beamforming
    def optimize_group_SNR(self, groups=None): # Groups: M x K
        if groups is None:
            group_list = np.ones((self.M, 1), dtype=bool)
            rev_idx = np.zeros((self.K), dtype=int) # Repeat the first element
        else:
            group_list, rev_idx = np.unique(groups, return_inverse=True, axis=1)
            
        group_rates = np.zeros((self.K))
        for i in range(group_list.shape[1]):
            group_ch = self.H[group_list[:, i], :]
            f_opt = Beamforming.iterative_BF(group_ch, self.cpu_phases)
            group_rates[i] = np.min(np.abs(group_ch @ f_opt.conj()))**2

        return group_rates[rev_idx]
    
    # Develop # Hybrid Beamforming solution
    # TODO: Implement MMSE and compute the rates
    def optimize_ZF_SNR(self, groups=None):
        if groups is None:
            group_list = np.ones((self.M, 1), dtype=bool)
            rev_idx = np.zeros((self.K), dtype=int) # Repeat the first element
        else:
            group_list, rev_idx = np.unique(groups, return_inverse=True, axis=1)
            
        H_her = self.H[np.any(group_list, axis=1), :].T.conj()
        H_pinv = np.linalg.pinv(H_her)
        #f = H_pinv / np.linalg.norm(H_pinv)
        ch_SNR = np.trace(H_pinv@H_pinv.T.conj())
        #print(ch_SNR)
        SNR = 1./np.abs(ch_SNR)
            
        return SNR 
        # Normalization:
        #F_RF*
    
    # TDMA time allocation for given rates and bandwidth - solution given in the paper - iterative
    def optimize_TDMA_paper(self, R_fh, B_fh, R_ac, B_ac, initial_step=0.5, epsilon = 1e-4):
        upper_bound = (B_ac*R_ac) / (B_fh*R_fh)
        eta = 0.0
        t_fn = lambda x: np.minimum(x / R_fh, upper_bound)
        current_sum = np.sum(t_fn(eta))
        step = initial_step
        sign = True
        # Step halving eta update
        while np.abs(current_sum-1)>epsilon:
            past_sum = current_sum
            if (sign and current_sum>1) or (not sign and current_sum<1):
                sign = not sign
                step /= 2
            
            if sign:
                eta += step
            else:
                eta -= step
            current_sum = np.sum(t_fn(eta))
            if current_sum == past_sum and current_sum<1:
                break
        return t_fn(eta)
    
    # TDMA time allocation for given rates and bandwidth - faster optimal solution
    def optimize_TDMA_fast(self, R_fh, B_fh, R_ac, B_ac):
        # Fast Solution
        max_vals = B_ac*R_ac/(B_fh)
        sorted = max_vals.argsort()
        inv_sort = sorted.argsort()
        
        R_fh = R_fh[sorted]
        max_vals = max_vals[sorted]
        
        t_opt = np.zeros(len(R_ac))
        for i in range(len(R_ac)):
            t_opt[i:]= max_vals[i]/R_fh[i:]
            if t_opt.sum()>1:
                t_opt[i:] = (1-np.sum(t_opt[:i]))/np.sum(1/R_fh[i:])/R_fh[i:] # Split remaining time 
                break
        
        return t_opt[inv_sort]
    
    # Equal time allocation
    def equal_TDMA(self, R_fh):
        return (1/R_fh)/np.sum(1/R_fh)
    
    def generate_fronthaul_channel(self, state, channel_fun):
        vars = np.array([[state.CPU.distance(AP), state.CPU.angle(AP)] for AP in (state.APs if state.wire_cluster_heads is None else [state.APs[i] for i in state.wire_cluster_heads])]) 
        # Pick only the channels of the cluster heads if available
        
        return channel_fun(dist=vars[:, 0], angle=vars[:, 1])
    
#%% Beamforming solutions
class Beamforming:
    
    @staticmethod
    def iterative_BF(H, cpu_phases, epsilon=0.00, max_iter=10000, random_trials=100):
        L = H.shape[1]
        f_opt = np.zeros(L, dtype=complex)
        rate_opt = 0
        
        # Try with different random initializations
        for _ in range(random_trials):
            f = np.random.choice(cpu_phases, L)
            iter = 0
            found = False
            while iter < max_iter and not found:
                
                f_p = f.copy()
                
                for l in range(L): # for each antenna
                    f_all = np.tile(f, (len(cpu_phases), 1))
                    f_all[:, l] = cpu_phases
                    group_rates = np.min(np.abs(H @ f_all.conj().T), 0)
                    l_opt = np.argmax(group_rates)
                    group_rate = group_rates[l_opt]
                    f[l] = cpu_phases[l_opt]
                
                found = np.sum(np.abs(f_p-f))<= epsilon # If no changes in the last iteration           
                iter += 1
            #print(iter)
            if group_rate > rate_opt:
                rate_opt = group_rate
                f_opt = f
                
        return f_opt/np.sqrt(L)