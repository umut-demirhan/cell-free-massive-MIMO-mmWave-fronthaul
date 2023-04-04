# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 07:12:46 2022

@author: Umt
"""
import numpy as np
import glob
import os
import pandas as pd
from tqdm import tqdm
import re

LIGHTSPEED = 299792458
BOLTZMANN = 1.3807e-23
T_NF = 290 
           
#%% Utilities
def path_loss_LOS(dist, fc, sigma_shadowing=4):
     return db2pow(-32.4 - 21*np.log10(dist) - 20*np.log10(fc/1e9) - np.random.standard_normal(len(dist))*sigma_shadowing)
 
def path_loss_NLOS(dist, fc, sigma_shadowing=8.2):
     return db2pow(-32.4 - 31.9*np.log10(dist) - 20*np.log10(fc/1e9) - np.random.standard_normal(len(dist))*sigma_shadowing)
 
def channel_accesslink(dist, ap_dist, ue_dist, **kwargs):
    fc = kwargs['fc']
    d_decorr = kwargs['shadowing']['d_decorr']
    delta = kwargs['shadowing']['delta']
    sigma_shadowing = kwargs['shadowing']['sigma_NLOS']
    
    M = dist.shape[0]
    K = dist.shape[1]

    a_corr = 2**(-ap_dist/d_decorr)
    b_corr = 2**(-ue_dist/d_decorr)
    
    a_low = np.linalg.cholesky(a_corr)
    b_low = np.linalg.cholesky(b_corr)
    
    a = (a_low @ np.random.standard_normal((M, 1)) ) * sigma_shadowing/np.sqrt(2)
    b = (b_low @ np.random.standard_normal((K, 1)) ) * sigma_shadowing/np.sqrt(2)
    
    a = np.tile(a.reshape((-1, 1)), (1, K))
    b = np.tile(b.reshape((1, -1)), (M, 1))
    
    shadowing = np.sqrt(delta)*a + np.sqrt(1-delta)*b
    return db2pow(-32.4 - 31.9*np.log10(dist) - 20*np.log10(fc/1e9) - shadowing)

def db2pow(db):
    return 10**(db/10.)
    
def array_response(angle, num_ant):
    return (1/np.sqrt(num_ant)) * np.exp(1j*np.pi*np.outer(np.cos(angle), np.arange(num_ant))); # Cos or Sin?

def analog_phase_shifts(bits):
    samples = 2**bits
    phase_set = np.exp(1j*2*np.pi*np.arange(samples) / samples)
    return phase_set

#%% Channels         
def channel_fronthaul_single_LOS_path(dist, angle, **kwargs):
    PL = path_loss_LOS(dist=dist, fc = kwargs['fc'])
    array_resp = array_response(angle=angle, num_ant=kwargs['L'])
    channel = np.sqrt(kwargs['L']) * np.sqrt(PL).reshape(-1, 1) * array_resp
    return channel
    
# def channel_accesslink(dist, angle, **kwargs):
#     fc = kwargs['fc']
#     ue_dist = kwargs['ue_dist']
#     PL = path_loss(dist=TX.distance(RX), fc=kwargs['fc'])
#     return PL

def channel_fronthaul_multi_path(dist, angle, **kwargs):
    fc = kwargs['fc']
    num_ant = kwargs['L']
    sf_sigma_LOS = kwargs['shadowing']['sigma_LOS']
    sf_sigma_NLOS = kwargs['shadowing']['sigma_NLOS']
    
    num_users = len(dist)
    PL_LOS = path_loss_LOS(dist = dist, fc = fc, sigma_shadowing = sf_sigma_LOS)
    array_resp_LOS = array_response(angle=angle, num_ant=num_ant)
    channel_LOS = np.sqrt(kwargs['L']) * np.sqrt(PL_LOS).reshape(-1, 1) * array_resp_LOS
    
    PL_NLOS = path_loss_NLOS(dist = dist, fc = fc, sigma_shadowing = sf_sigma_NLOS)
    user_paths = np.random.randint(1, 7, num_users) # 1-6 random paths
    num_paths = user_paths.sum()
    rayleigh_NLOS = (1/np.sqrt(2)) * (np.random.standard_normal(size=num_paths) + 1j*np.random.standard_normal(size=num_paths))
    angle_NLOS = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=num_paths) # uniform in [-pi/2, pi/2)
    array_resp_NLOS = array_response(angle=angle_NLOS, num_ant=num_ant)
    channel_path_NLOS = rayleigh_NLOS.reshape(-1, 1) * np.repeat(PL_NLOS, user_paths).reshape(-1, 1) * array_resp_NLOS
    channel_NLOS = np.zeros((num_users, num_ant), dtype=np.cdouble)
    start = 0
    for u in range(num_users):
        channel_NLOS[u, :] = channel_path_NLOS[start:user_paths[u], :].sum(axis=0)
        start = start + user_paths[u]
    channel = channel_LOS + channel_NLOS
    
    return channel

#%%
def load_files(directory, params, load_num = 3):
    
    # Generate a table of result files
    filelist = glob.glob(os.path.join(os.path.abspath(directory), '*'))
    file_frame =pd.DataFrame()
    for i in tqdm(range(len(filelist))):
        # Generate a file list
        file_dict = {'Path': filelist[i]}
        vals = os.path.basename(filelist[i]).split('.')[0].split('_') # First file
        for i in range(len(vals)):
            datatuple = re.findall(r'([a-z A-Z]+)([0-9]+)', vals[i])[0]
            file_dict[datatuple[0]] = [int(datatuple[1])]
        file_table = pd.DataFrame.from_dict(file_dict, orient='columns')
        file_frame = file_frame.append(file_table, ignore_index=True)
    
    # Check fully generated repetitions
    num_files_per_rep = (file_frame['Rep']==0).sum() # Find number of files from index 0
    unique_reps = file_frame['Rep'].unique()
    full_reps = []
    for i in unique_reps:
        if (file_frame['Rep']==i).sum() == num_files_per_rep:
            full_reps.append(i)
        else:
            file_frame = file_frame[(file_frame['Rep']!=i)] # Remove if not fully available
            print('Repetition %i is not fully available, will not be loaded..'%i)
    full_reps = np.array(full_reps)
    full_reps.sort()
    file_frame = file_frame.reset_index(drop=True)
    cols = list(file_frame.columns.copy())
    cols.remove('Path')
    file_frame = file_frame.sort_values(by=cols, axis=0, ignore_index=True)
    
    # Generate array for data and corresponding index mapping
    ignored_cols = ['Path']
    col_order = []
    col_order_new = []
    col_map = {}
    data_size = []
    for column in file_frame.columns:
        if column not in ignored_cols:
            column_new = column + '_idx'
            col_order.append(column)
            col_order_new.append(column_new)
            unique_vals = file_frame[column].unique()
            unique_vals_reset = np.arange(len(unique_vals))
            col_map[column] = np.array([unique_vals_reset, unique_vals])
            
            column_new_vals = list(map(lambda x: unique_vals_reset[np.where(unique_vals==x)][0], file_frame[column]))
            file_frame = pd.concat([file_frame, pd.DataFrame(column_new_vals, columns=[column_new])], axis=1)
            data_size.append(len(unique_vals))
            
    
    data_size.insert(0, load_num)
    data = np.zeros(tuple(data_size))

    for i in tqdm(range(len(file_frame))):
        filename = file_frame['Path'][i]
        values = np.load(filename)
        for j in range(load_num):
            data[(j,)+tuple(file_frame.iloc[i][col_order_new])] =  values['arr_%i'%j].sum()
            
    return data, col_order, col_map