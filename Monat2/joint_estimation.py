#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:21:55 2017

@author: Liwen
"""
import numpy as np
from f_sync import ML_approx_unknown
from commpy.utilities import bitarray2dec 

class joint_estimation():
    def __init__(self):
        pass
    def function(self,r_syc_coarse,N,N_known,T,ibits_known,symbols_known,SA,RA):
        n_range=N-N_known
        L=np.zeros(n_range-1)
        #when n=0 function doesn't work,this situation should be tested by a if command
        index=bitarray2dec(ibits_known)
        for n in range(1,n_range):   
            f_estt=ML_approx_unknown(r_syc_coarse[n:n+N_known],T,symbols_known,ibits_known)
            H_est=np.zeros([RA,SA],complex)  
            i=np.zeros(SA)
            #Channel estimation
            r=r_syc_coarse[n:n+N_known+1]
            for k in range(0,symbols_known.size):
                H_est[:,index[k]]+= r[k,:]/symbols_known[k]*np.exp(-1j*2*np.pi*T*f_estt*(k+n))
                i[index[k]]=i[index[k]]+1
            #Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden
            H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
           
            #Likelihood function for Timing estimation       
            for m in range(0,symbols_known.size):
                L[n-1]+=np.linalg.norm(r[m,:]-H_est[:,index[m]]*np.exp(1j*2*np.pi*T*m)*symbols_known[m])**2
            n_est=np.argmin(L)+1   
        
        r=r_syc_coarse[n_est:n_est+N_known]
        
        ##one more estimation after estimation for n (or save Estimation results for each n ?
        f_est=ML_approx_unknown(r,T,symbols_known,ibits_known)
        H_est=np.zeros([RA,SA],complex) 
        i=np.zeros(SA)
        for k in range(0,symbols_known.size):
            H_est[:,index[k]]+= r[k,:]/symbols_known[k]*np.exp(-1j*2*np.pi*T*f_est*(k+n_est))
            i[index[k]]=i[index[k]]+1
        self.H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
        self.f_est=f_est
        self.n_est=n_est
    