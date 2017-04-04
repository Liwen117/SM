#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:17:17 2017

@author: lena
"""
import numpy as np

def rint(x):
    return np.int(np.round(x))

class gardner_timing_recovery:
    e = [0]
    gamma = 1e-1
    tau = [0, 0]
    output_symbols = []
    
    def run(self, y):
        for k in range(y.size//sps - 1):
            self.output_symbols.append(y[k*sps + rint(self.tau[k])])
            if k > 0:
                self.e.append(self.TED(y, k))  # update error signal
                self.tau.append(self.loop_filter(k))
    
    def TED(self, y, k):
        return (y[(k-1)*sps + self.rint(self.tau[k-1])] - y[k*sps + rint(self.tau[k])]) * y[k * sps - sps//2 + rint(self.tau[k-1])]
    
    def loop_filter(self, k):
        return self.tau[k] + self.gamma * self.e[k]
    
    def rint(self, x):
        return int(round(x))
    
timing_sync = gardner_timing_recovery()
timing_sync.run(r)
r_sync = timing_sync.output_symbols