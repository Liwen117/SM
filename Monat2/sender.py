#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:56:21 2017

@author: Liwen
"""
import numpy as np  
from commpy.utilities import bitarray2dec
import training_sequence as tr

class sender():
    def __init__(self,N_simu,N,Ni,Nd,mapp,filter_,k):
        self.N_simu=N_simu
        self.N=N
        self.Ni=Ni
        self.Nd=Nd
        self.mapp=mapp
        self.ir=filter_.ir()
        self.sps=filter_.n_up
        self.index_SA=0
        self.k=k
        self.Ni_est=1
#        self.Ni_est=2**Ni
        #self.ibits,self.dbits=self.generate_simu_bits(N_simu,N,Nd,Ni)
        #self.idbits=np.random.choice([0,1],self.N*(self.Ni+self.Nd))     
#        self.n_start=17

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    def send(self):
            self.generate_simu_bits(self.N_simu,self.N,self.Nd,self.Ni,self.k,self.mapp)  
            self.bbsignal()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def generate_random_ibits(self):        
        return np.random.choice([0,1],self.N_simu*self.Ni) 
    
    def generate_random_dbits(self):
        return np.random.choice([0,1],self.N_simu*self.Nd)

    def generate_preambel(self,N,Ni,Nd,k,mapp):
#        self.dbits_known=tr.ts_d(self.N,self.k,self.Nd)
        self.symbols_known=[]
        self.ibits_known=[]
        for index_SA in range(0,self.Ni_est):
            self.symbols_known=np.concatenate((self.symbols_known,tr.ts_d(self.N,self.k,self.mapp)))
#            print(tr.ts_i(self.N, self.Ni,0).shape,tr.ts_i(self.N, self.Ni,1).shape)
            self.ibits_known=np.append(self.ibits_known,tr.ts_i(self.N, self.Ni,index_SA))
        return self.ibits_known,self.symbols_known
        
    def generate_simu_bits(self,N_simu,N,Nd,Ni,k,mapp):  
        self.generate_preambel(N,Ni,Nd,k,mapp)
        self.ibits=np.concatenate((self.generate_random_ibits().reshape((Ni,-1)),self.ibits_known.reshape((Ni,-1)),self.generate_random_ibits().reshape((Ni,-1))),1 ) 
#        self.dbits=np.concatenate((self.generate_random_dbits(N_simu,Nd).reshape((Nd,-1)),self.dbits_known.reshape((Nd,-1))),1)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    def divide_index_data_bits(self):
#        #if (idbits.size % (Ni+Nd) !=0):
#        #   idbits=idbit[:idbits.size-idbits.size % (Ni+Nd)]
#        divided_bits=self.idbits.reshape((self.Nd+self.Ni,-1))
#        ibits=divided_bits[0:self.Ni,:]
#        dbits=divided_bits[self.Ni:self.Ni+self.Nd,:]
#        self.dbits= dbits
#        self.ibits= ibits
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    def databits_mapping(self,Nd,N_simu,N):
        indices1=bitarray2dec(self.generate_random_dbits().reshape((Nd,-1)))
        indices2=bitarray2dec(self.generate_random_dbits().reshape((Nd,-1)))
#        self.symbols_known=self.mapp[bitarray2dec(self.dbits_known.reshape((Nd,-1))) ]
        self.symbols=np.concatenate((self.mapp[indices1],self.symbols_known,self.mapp[indices2]))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    def databits_pulseforming(self,symbols):
        #repeat value
        s = np.zeros(symbols.size*self.sps+self.ir.size-1)
        for i in range(symbols.size):
            s[i*self.sps:i*self.sps+self.ir.size]+=symbols[i]*self.ir
         #??? which one is right/better???
         #zero-padding
#        symbols_up = np.zeros(symbols.size * self.sps)
#        symbols_up[::self.sps] = symbols
#        s = np.convolve(self.ir, symbols_up)            
        return s
      
#        symbols_up = np.repeat(symbs,self.sps)
#        return np.convolve(self.ir, symbols_up)
            
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    def bbsignal(self):
        #self.divide_index_data_bits()
        group_delay = (self.ir.size - 1) // 2
        self.databits_mapping(self.Nd,self.N_simu,self.N)
        s_BBr=self.databits_pulseforming(np.real(self.symbols))
        s_BBi=self.databits_pulseforming(np.imag(self.symbols))
#        return s_BBr+1j*s_BBi
        return s_BBr[group_delay:-group_delay]+1j*s_BBi[group_delay:-group_delay]

    
    
    
    