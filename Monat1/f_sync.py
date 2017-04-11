#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
from commpy.utilities import bitarray2dec 

#data-aided ML Approximation with known channel(Voraussetzung:f_off<<1/T)
def ML_approx_known(r_,T,symbols,ibits,H):
    f_delta=1
    f_range=100
    interp_fact=10
    p=np.zeros([int(f_range/f_delta),H.shape[0]],complex)
    f_o=np.zeros(H.shape[0])
    xvals = np.linspace(0, f_range, f_range/f_delta*interp_fact)
    x = np.linspace(0, int(f_range/f_delta),int(f_range/f_delta))
    for j in range(0,H.shape[0]):
        s_a_index=bitarray2dec(ibits)
        for f in range(0,int(f_range/f_delta)):
            for i in range(0,r_.shape[0]):
                off_=np.exp(-1j*2*np.pi*f*f_delta*T*i)
                p[f,j]+=symbols[i]*r_[i,j]*H[j,s_a_index[i]]*off_
        #with Interpolation            
        f_o[j]=np.argmax(np.interp(xvals,x,np.abs(p[:,j])))*f_delta**2/interp_fact
    return f_o
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#non data-aided, MPSK

#def NDA(r,M,T):
#    summ=np.zeros(r.shape[1],complex)
#    f_off=np.zeros(r.shape[1])
#    for j in range(0,r.shape[1]):
#        for i in range(1,r.shape[0]):
#            summ[j] += (r[i,j]*np.conj(r[i-1,j]))**M
#            f_off[j]=1/(2*np.pi*T*M)*(np.angle(summ[j]))
#    return f_off

def NDA(r,M,T,H): 
    summ=np.zeros([r.shape[1],H.shape[1],H.shape[1]],complex)
    f_off=np.zeros([r.shape[1],H.shape[1],H.shape[1]])
    for j1 in range(0,H.shape[1]):
       for j2 in range(0,H.shape[1]):
           for i in range(0,r.shape[1]):
               for n in range(1,r.shape[0]):
                   summ[i,j1,j2] += (r[n,i]*H[i,j1]*np.conj(r[n-1,i]*H[i,j2]))**M
                   f_off[i,j1,j2]=1/(2*np.pi*T*M)*(np.angle(summ[i,j1,j2]))
    return f_off
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#data-aided ML with channel unknown
def ML_unknown(y,T,symbols,ibits):
    f_delta=10
    f_range=10000
    interp_fact=1
    R=np.zeros(y.shape[0],complex)
    X=np.zeros(int(f_range/f_delta),complex)
    
    N=1/np.sum(symbols**2)
    index=bitarray2dec(ibits)
    xvals = np.linspace(0, f_range, f_range/f_delta*interp_fact)
    x = np.linspace(0, int(f_range/f_delta),int(f_range/f_delta))
    for f in range(0,int(f_range/f_delta)):
        for m in range(1,y.shape[0]):
            for k in range(m,y.shape[0]):
                if index[k]==index[k-m]:
                    R[m]+= np.dot(np.conj(y[k-m,:]),y[k,:])*symbols[k]*symbols[k-m]*N
            X[f] +=R[m]*np.exp(-1j*2*np.pi*m*f*f_delta*T)
        R[m]=0
    f_est=np.argmax(np.interp(xvals,x,np.real(X)))*f_delta**2/interp_fact
    return f_est
            

            


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#data-aided ML Approximation with channel unknown
def ML_approx_unknown(y,T,symbols,ibits):

    R=np.zeros(y.shape[0],complex)

    X_1=0
    X_2=0
    N=1/np.sum(symbols**2)
    index=bitarray2dec(ibits)
    
    for m in range(0,y.shape[0]):
        for k in range(m,y.shape[0]):
            if index[k]==index[k-m]:
                R[m]+=np.dot(np.conj(y[k-m,:]),y[k,:])*symbols[k]*symbols[k-m]*N
        X_1 +=m*np.abs(R[m])*np.angle(R[m])
        X_2 +=m**2*np.abs(R[m])
    
    f_est=1/(2*np.pi*T)*X_1/X_2

    return f_est
            

#braucht ggf. Anpassung auf Anzahl der Benutzung von SA=> mit steigender N verschlechtet sich die Schätzung






