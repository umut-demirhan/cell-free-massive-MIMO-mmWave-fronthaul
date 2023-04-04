# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 00:04:01 2022

@author: Umt
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Object placement
def uniform_square_rand_object(n, edge=100, center=np.array([0, 0])):
    objects = []
    for i in range(n):
        location = np.random.uniform(low = center - edge/2, high = center + edge/2)
        objects.append(CommObject(location))
    return objects

def organized_square_object(n, edge=100, center=np.array([0, 0])):
    objects = []
    n_sqrt = int(np.sqrt(n))
    dist = edge/n_sqrt/2
    positions_single_ax = np.linspace(dist, edge-dist, n_sqrt) - edge/2
    positions = np.array(np.meshgrid(positions_single_ax, positions_single_ax)).reshape(2, n) - center.reshape(-1, 1)
    for i in range(n):
        location = positions[:, i]
        objects.append(CommObject(location))
    return objects

#%% State - Main Object for instanciating
class State():
    
    def __init__(self, cpu_loc, M, K, L, edge, center, fn_UE=uniform_square_rand_object, fn_AP=uniform_square_rand_object, wire_clusters=None, wire_cluster_heads=None):
        self.COLORS = [['b', 'r'], ['orange', 'green']]
        
        self.UEs = CommObjectContainer(fn_UE(n=K, edge=edge, center=center))
        self.APs = CommObjectContainer(fn_AP(n=M, edge=edge, center=center))
        self.CPU = CommObject(location=cpu_loc)

        self.wire_clusters = wire_clusters
        self.wire_cluster_heads = wire_cluster_heads
        
    def plot(self, AP_idx=None, UE_idx=None, plot_AP=True, plot_UE=True, plot_CPU=True, color_scheme=0):
        AP_locations = self.APs.locations()
        UE_locations = self.UEs.locations()
        plot_AP_idx = np.arange(len(AP_locations)) if AP_idx is None else [AP_idx] if isinstance(AP_idx, int) else AP_idx # Plots all if None, or takes single value, or list
        plot_UE_idx = np.arange(len(UE_locations)) if UE_idx is None else [UE_idx] if isinstance(UE_idx, int) else UE_idx # Plots all if None, or takes single value, or list
        
        if plot_AP:
            if self.wire_cluster_heads is None: # All Circle APs
                plt.scatter(AP_locations[plot_AP_idx, 0], AP_locations[plot_AP_idx, 1], marker='o', facecolors='none', edgecolors=self.COLORS[color_scheme][0], label='AP')
            else: # Cluster Heads Square - The others circle
                cluster_elements_plot = np.intersect1d(plot_AP_idx, self.wire_cluster_heads)
                plt.scatter(AP_locations[cluster_elements_plot, 0], AP_locations[cluster_elements_plot, 1], marker='s', facecolors='none', edgecolors=self.COLORS[color_scheme][0], label='AP')
                others = np.setdiff1d(plot_AP_idx, self.wire_cluster_heads)
                plt.scatter(AP_locations[others, 0], AP_locations[others, 1], marker='o', facecolors='none', edgecolors=self.COLORS[color_scheme][0], label='AP')
        if plot_UE:
            plt.scatter(UE_locations[plot_UE_idx, 0], UE_locations[plot_UE_idx, 1], marker='x', c=self.COLORS[color_scheme][1], label='UE')
        if plot_CPU:
            plt.scatter(self.CPU.location[0], self.CPU.location[1], marker='s', c='k', label='CPU')
            
        if self.wire_clusters is not None: # Wire connections
            for cluster_idx in range(len(self.wire_clusters)):
                plot_wire = np.intersect1d(plot_AP_idx, self.wire_clusters[cluster_idx])
                plt.plot(AP_locations[plot_wire, 0], AP_locations[plot_wire, 1], color=self.COLORS[color_scheme][0], linestyle='-')
        plt.legend()
        
    def plot_group(self, UE_idx, ac_groups):
        self.plot()
        self.plot(AP_idx=np.where(ac_groups[:, UE_idx]) , UE_idx = UE_idx, color_scheme=1)
        ax = plt.gca()
        ax.get_legend().remove()

#%% Comm Objects
class CommObjectContainer():
    def __init__(self, object_list):
        self.objects = object_list

    def __getitem__(self, key):
        return self.objects[key]

    def __setitem__(self, key, value):
        self.objects[key] = value

    def __len__(self):
        return len(self.objects)

    def locations(self):
        return np.array([object.location for object in self.objects])

class CommObject():
    def __init__(self, location):
        self.location = location
        
    def distance(self, comm_element):
        return np.sqrt(np.sum(np.abs(self.location - comm_element.location)**2))
    
    def angle(self, comm_element):
        dist_vec = comm_element.location - self.location
        return np.arctan2(dist_vec[0], dist_vec[1])
            