# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 07:12:13 2022

@author: Umt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from fronthaul import Fronthaul
from accesschannel import AccessChannel
from sim_objects import State
from functools import partial
import grouping
from utilities import *
    
def sample(rep_idx, params, save_folder='fig4'):
    print(rep_idx)
    for M_idx in range(len(params['M'])):
        M = params['M'][M_idx]
        state = State(cpu_loc=params['cpu_loc'], 
                          M=M, 
                          K=params['K'], 
                          L=params['L'], 
                          edge=params['edge'], 
                          center=params['center'])       
            
        # Define Fronthaul
        
        # Wrap channel function to only take TX and RX objects as input
        fh_channel_fun1 = partial(channel_fronthaul_multi_path, L=params['L'], fc=params['fc_fh'], shadowing=params['shadowing'])
            
        fh1 = Fronthaul(state=state, 
                           channel_fun=fh_channel_fun1,  
                           cpu_ant=params['L'],
                           cpu_phases=params['phase_set'],
                           num_groups=params['K'])
        
        # fh_channel_fun2 = partial(channel_fronthaul_single_LOS_path, L=params['L'], fc=params['fc_fh'])
        
        # fh2 = Fronthaul(state=state, 
        #                channel_fun=fh_channel_fun2,  
        #                cpu_ant=params['L'],
        #                cpu_phases=params['phase_set'],
        #                num_groups=params['K'])
        
        fh_list = [fh1]
        
        # Define Access Channel
        
        # Wrap channel function to only take TX and RX objects as input
        ac_channel_fun = partial(channel_accesslink, fc=params['fc_ac'], shadowing=params['shadowing'])
        
        ac = AccessChannel(state=state,
                           channel_fun=ac_channel_fun,
                           ap_power=params['P_norm_AP'])
            
        # Calculate Rates 
        # Fronthaul
        channel_est = ac.channel_estimation()
        for g in range(len(params['G'])):
            #print('Group size %i'% params['G'][g])
            groups = grouping.power_based_fixed_groups(channel_est, group_size=params['G'][g])
            R_ac = ac.rate_optimal_power(channel_est, groups=groups)
            
            for fh_idx in range(len(fh_list)):
                fh = fh_list[fh_idx]
                SNR_fh = fh.optimize_group_SNR(groups=groups) # Optimize BF Fronthaul
                for bw in  range(len(params['B_fh'])): # Fronthaul BW
                    bandwidth = params['B_fh'][bw]
                    #print('   BW %.2f (MHz)'% (bandwidth/1.e6))
                    params['P_norm_CPU'] = params['p_CPU'] / (params['sigma_fh'] * bandwidth * BOLTZMANN * T_NF)
                    R_fh = np.log2(1 + params['P_norm_CPU']*SNR_fh)
                    t_eq = fh.equal_TDMA(R_fh)
                    t_opt = fh.optimize_TDMA_fast(R_fh, bandwidth, R_ac, params['B_ac'])
                    fh_rates = R_fh*t_opt*bandwidth/1e6
                    ac_rates = R_ac*params['B_ac']/1e6
                    fh_equal_rates = R_fh*t_eq*bandwidth/1e6
                    save_loc = os.path.abspath(os.path.join(save_folder, 'Rep%i_FH%i_M%i_BW%i_G%i.npz'%(rep_idx, fh_idx, M_idx, bw, g)))
                    np.savez(save_loc, *[fh_rates, ac_rates, fh_equal_rates, groups, R_ac, R_fh])
                    #print('AC rate: %.2f, FH Rate: %.2f'%(ac_rates.sum(), fh_rates.sum()))
                    #print(ac_rates-fh_rates)
        

#%% Run
if __name__=='__main__':
    
    params =     {'reps': 10, # Repetition of this simulation
                  'M': np.arange(30, 91, 10), # Number of APs
                  'K': 10, # Number of UEs
                  'L': 128, # Number of CPU antennas
                  
                  'G': np.array([30, 40, 50]), # Group size, fully select if None
                  
                  'p_CPU': db2pow(0), # CPU Pow: 30 dBm
                  'p_AP': db2pow(-20), # AP Pow: 10 dBm
                  
                  'fc_fh': 28e9, # FH Carrier Freq.: 28 GHz
                  'fc_ac': 3.5e9, # AC Carrier Freq.: 3.5GHz
                      
                  'B_fh': np.array([120e6, 240e6]), # Fronthaul BW: 1 GHz
                  'B_ac': 20e6, # AC BW: 20 MHz
                  
                  'sigma_fh': db2pow(9), # FH Noise Figure
                  'sigma_ac': db2pow(9), # AC Noise Figure
                  
                  'shadowing': 
                      {'delta': 0.5, 
                       'd_decorr': 100,
                       'sigma_LOS': 4, # dB
                       'sigma_NLOS': 8.2 # dB
                       },
                      
                  'q_CPU': 3, # Beamforming quantization
                  # Area Settings
                  'center': np.array([0, 0]),
                  'edge' : 100,
                  'cpu_loc': np.array([-100, 0])
                  }
    add_params = {'lambda_fh': LIGHTSPEED/params['fc_fh'],
                  'lambda_ac': LIGHTSPEED/params['fc_ac'],
                  
                  'P_norm_CPU': params['p_CPU'] / (params['sigma_fh'] * params['B_fh'] * BOLTZMANN * T_NF),
                  'P_norm_AP': params['p_AP'] / (params['sigma_ac'] * params['B_ac'] * BOLTZMANN * T_NF),
                  
                  'phase_set': analog_phase_shifts(params['q_CPU'])
                  }
    params = dict(params, **add_params)
    
    folder = 'fig_AP_scaling'
    processes = 16
    
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    sample_par = partial(sample, params=params, save_folder=folder)
    
    # #sample_par(1)
    
    # # %% Multi-proc. solution
    pool = multiprocessing.Pool(processes=processes)
    pool.map(sample_par, range(params['reps']))
    pool.close()
    pool.join() 
    print('done')
    
    #%%
    # Load and plot files
    data, col_order, col_map = load_files(directory=folder, params=params)
    rep_pos = np.where([col == 'Rep' for col in col_order])[0][0]+1
    data = data.mean(axis = rep_pos)
    
    #%%
    fh_select = 0
    new_map = col_map['M'][1].copy()
    data_plot = data[:, fh_select, new_map, :]
    markers = ['*', 'x', 's', 'd', 'o', '^']
    from matplotlib.pyplot import cm
    plt.figure()
    list_BW = [0, 1]
    list_G = [0, 1, 2]
    colors = plt.get_cmap("tab10")#iter(cm.jet(np.linspace(0, 1, 3)))
    for j in range(len(list_G)):
        color = colors(j)#next(colors)
        j = list_G[j]
        for i in range(len(list_BW)):
            marker = markers[i]
            i = list_BW[i]
            plt.plot(params['M'], data_plot[2, :, i, j], '--', label='FH-%iMHz, G=%i'%(int(params['B_fh'][i]/1e6), params['G'][j]), c=color, marker=marker)
            plt.plot(params['M'], data_plot[0, :, i, j], '-.', label='E2E-%iMHz'%int(params['B_fh'][i]/1e6), c=color, marker='o')
        
        plt.plot(params['M'], data_plot[1, :, 0, j], '-', label='Access Channel, G=%i'%(params['G'][j]), c=color)
        
    plt.xlabel('Number of APs (M)')
    plt.ylabel('Sum Rate (Mbps)')
    plt.legend()
    plt.grid()
        
    #%%
    import scipy.io
    plot_dic = {'data_plot': data_plot,
                'col_map': col_map,
                'params': params}
    scipy.io.savemat('mat_files/' + folder+'.mat', plot_dic)
        
        
