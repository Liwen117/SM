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
        #self.ibits,self.dbits=self.generate_simu_bits(N_simu,N,Nd,Ni)
        #self.idbits=np.random.choice([0,1],self.N*(self.Ni+self.Nd))
        self.n_start=np.random.randint(0,N_simu-N)      
#        self.n_start=17
        self.generate_simu_bits(N_simu,N,Nd,Ni,k,mapp)
        self.bbsignal()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def generate_random_ibits(self,N,Ni):        
        return np.random.choice([0,1],N*Ni) 
    
    def generate_random_dbits(self,N,Nd):
        return np.random.choice([0,1],N*Nd)

    def generate_preambel(self,N,Ni,Nd,k,mapp):
        self.ibits_known,self.symbols_known=tr.sc(N,Ni,k,Nd,self.index_SA)
        self.index_SA+=1
        
    def generate_simu_bits(self,N_simu,N,Nd,Ni,k,mapp):
        
        self.ibits=np.concatenate((self.generate_random_ibits(N_simu-N,Ni).reshape((Ni,-1)),self.ibits_known,np.random.choice([0,1],(N_simu-self.n_start-N)*Ni).reshape((Ni,-1))),1 ) 
        self.dbits=np.concatenate()
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
        indices1=bitarray2dec(np.random.choice([0,1],self.n_start*Nd).reshape((Nd,-1)))
        indices2=bitarray2dec(np.random.choice([0,1],(N_simu-self.n_start-N)*Nd).reshape((Nd,-1)))
        #self.symbols_known=self.mapp[bitarray2dec(self.dbits_known) ]
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
    def only_upsampling(self):
#repeat value
        symbols_up = np.repeat(self.symbols,self.sps)
         #??? which one is right/better???
         #zero-padding
#        symbols_up = np.zeros(self.symbols.size * self.sps)
#        symbols_up[::self.sps] = self.symbols
        self.symbols_up=symbols_up
        return symbols_up

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        

    def bbsignal(self):
        #self.divide_index_data_bits()
        group_delay = (self.ir.size - 1) // 2
        self.databits_mapping(self.Nd,self.N_simu,self.N)
        s_BBr=self.databits_pulseforming(np.real(self.symbols))
        s_BBi=self.databits_pulseforming(np.imag(self.symbols))
#        return s_BBr+1j*s_BBi
        return s_BBr[group_delay:-group_delay]+1j*s_BBi[group_delay:-group_delay]
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#    def anti_image(self,signal):
#        SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal)))**2/signal.size
#        return SIGNAL
#    
    
    
    
    
    