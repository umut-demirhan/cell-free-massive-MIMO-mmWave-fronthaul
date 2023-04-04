# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 05:45:25 2022

@author: Umt
"""
import numpy as np

def power_based_fixed_groups(channel_est, group_size): # M by K
    if group_size is None:
        group_size = channel_est.shape[0]
        
    ind = np.argsort(channel_est, axis=0)
    groups = np.zeros(channel_est.shape, dtype=bool)
    np.put_along_axis(groups, ind[-group_size:, :], True, axis=0)
    return groups