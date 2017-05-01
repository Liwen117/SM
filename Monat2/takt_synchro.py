#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:36:17 2017

@author: Liwen
"""
import numpy as np  # make the numpy package available and use 'np' as alias
import scipy.linalg as lin
import matplotlib.pyplot as plt  # plotting library


class gardner_timing_recovery:
    e = [0]
    gamma = 1e-1
    tau = [0, 0]
    output_symbols = []
    def __init__(self, n_up):
        self.n_up=n_up
        
    def run(self, y):
        for k in range(y.size//self.n_up - 1):
            self.output_symbols.append(y[k*self.n_up + np.rint(self.tau[k])])
            if k > 0:
                self.e.append(self.TED(y, k))  # update error signal
                self.tau.append(self.loop_filter(k))
    
    def TED(self, y, k):
        return (y[(k-1)*self.n_up + np.rint(self.tau[k-1])] - y[k*self.n_up + np.rint(self.tau[k])]) * y[k * self.n_up - self.n_up//2 + np.rint(self.tau[k-1])]
    
    def loop_filter(self, k):
        return self.tau[k] + self.gamma * self.e[k]
    
    def rint(self, x):
        return int(round(x))
    ##Kommentar: Gardner funktioniert nicht bei SM (bzw. nur wenn Kanalkoeffiziente aehnlich sind)
    
def feedforward_timing_sync(n_up,y,m,N_known):
    takt=np.zeros(n_up)
    y=y[n_up*m:n_up*(m+N_known)]
    for i in range(0,y.size-2):
        if(np.abs( np.real(y[i+2]-y[i]))< np.abs(np.real(y[i+1]-y[i]))):
           takt[np.mod(i,n_up)]+=1
    print("takt_est=",n_up-1-np.argmax(takt))
    return n_up-1-np.argmax(takt)
  