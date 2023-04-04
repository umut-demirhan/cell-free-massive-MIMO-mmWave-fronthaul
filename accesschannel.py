# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 05:16:41 2022

@author: Umt
"""
import numpy as np
import cvxpy as cp

class AccessChannel():
    def __init__(self, state, channel_fun, ap_power):
        self.ap_power = ap_power
        self.H = self.generate_access_channel(state, channel_fun) # M x K (outer array is APs)
        self.K = len(state.UEs)
        self.M = len(state.APs)
    
    def channel_estimation(self, num_pilots=None, pilot_power=None):
        if num_pilots is None:
            num_pilots = self.K
        if pilot_power is None:
            pilot_power = self.ap_power
            
        beta_hat = (pilot_power * num_pilots * self.H**2) / (1 + pilot_power * num_pilots * self.H)
        
        return beta_hat
        

    def rate_optimal_power(self, beta_hat, groups=None, x_tol=1e-3):
        if groups is None:
            groups = np.ones(beta_hat.shape, dtype=bool)
        
        eta = cp.Variable((self.M, self.K), nonneg=True)
        v = cp.Variable((self.M), nonneg=True)
        t = cp.Parameter() # Feasibility region
        
        constraints = []
        for k in range(self.K):
            constraints.append(cp.SOC( np.sqrt(self.ap_power)*(1/cp.sqrt(t))*beta_hat[:, k] @ eta[:, k] , cp.hstack([np.sqrt(self.ap_power)* cp.multiply(np.sqrt(self.H[:, k]), v), 1]) ))
        for m in range(self.M):
            constraints.append(cp.SOC( v[m], cp.multiply(np.sqrt(beta_hat[m, :]), eta[m, :])))
        constraints.append(v <= 1)
        if (~groups).sum()>0:
            constraints.append(eta[~groups] == 0)
        
        problem = cp.Problem(cp.Minimize(0), constraints)
        
        def solve(t_val):
            t.value = t_val
            try:
                problem.solve(solver=cp.MOSEK)
            except:
                return False
            return problem.status == 'optimal'
        
        ###
        # Bisection
        iterations = 0
        eq_SINR = self.rate_equal_power(beta_hat, SINR=True)
        
        bisect_min = eq_SINR.min()
        bisect_max = eq_SINR.max()
        
        while not solve(bisect_min):
            iterations += 1
            bisect_max = bisect_min
            bisect_min /= 2
        while solve(bisect_max):
            iterations += 1
            bisect_min = bisect_max
            bisect_max *= 2
            
        while (bisect_max - bisect_min)>x_tol:
            iterations += 1
            t_test = (bisect_min + bisect_max)/2
            if solve(t_test):
                eta_opt = eta.value
                bisect_min = t_test
            else:
                bisect_max = t_test
                
        #print('Access Channel optimization is completed in %i bisection iterations..' % iterations)
        ###
        
        SINR = self.SINR(eta_opt, beta_hat)
        rate = np.log2(1+SINR)
        return rate
        
    def rate_equal_power(self, beta_hat, SINR=False):
        eta = np.tile(1/self.H.sum(axis=1), (self.K, 1)).T # Sum over each AP
        SINR_val = self.SINR(eta, beta_hat)
        rate = np.log2(1+SINR_val)
        return rate if not SINR else SINR_val
        
    def SINR(self, eta, beta_hat):
        nom = (self.ap_power * np.sum(beta_hat * eta, axis=0)**2)
        denom = (1 + self.ap_power * np.sum((eta**2) * beta_hat, axis=1) @ self.H)
        return nom/denom
    
    def generate_access_channel(self, state, channel_fun):
        
        dist = np.array([[state.APs[j].distance(state.UEs[i]) for i in range(len(state.UEs))] for j in range(len(state.APs))]) # M x K
        ap_dist = np.array([[state.APs[j].distance(state.APs[i]) for i in range(len(state.APs))] for j in range(len(state.APs))]) # M x M
        ue_dist = np.array([[state.UEs[j].distance(state.UEs[i]) for i in range(len(state.UEs))] for j in range(len(state.UEs))]) # K x K
        
        return channel_fun(dist, ap_dist, ue_dist)
        