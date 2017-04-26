#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
from commpy.utilities import bitarray2dec 
import scipy.linalg as lin
import scipy.signal as sig
import matplotlib.pyplot as plt 
#data-aided ML Approximation with known channel(Voraussetzung:f_off<<1/T) for each antenna
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

#data-aided ML with channel unknown
def ML_unknown(y,T,symbols,ibits):
    f_delta=1
    f_range=100
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
            
#non data-aided, MPSK 
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def NDA(r,M,T,H,n_up):    
    summ=np.zeros([r.shape[1],H.shape[1]],complex) 
    f_off=np.zeros([r.shape[1],H.shape[1]]) 
    for j in range(0,H.shape[1]): 
        for i in range(0,r.shape[1]): 
            for n in range(1,r.shape[0]): 
                summ[i,j] += (r[n,i]/H[i,j]*np.conj(r[n-1,i]/H[i,j]))**M 
                f_off[i,j]=1/(2*np.pi*T*M)*(np.angle(summ[i,j])) 
    return f_off*n_up
#kommentar: funktioniert nur bei sehr hohem SNR


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FLL(): 
    def __init__(self,g_mf,n_up):
        k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1) 
        T=1
 #Schrittweite 
        self.gamma = 0.01 
        self.g_mf=g_mf
        self.n_up=n_up
        self.g_dmf= 2*np.pi*T*k[:]*g_mf[:] 
        self.x_buf = np.zeros([len(g_mf)-1],complex) 
        self.x_1 = complex(0)
        self.y_buf = np.zeros([len(g_mf)-1],complex)
        self.y_1 = complex(0)
        self.nu = 0 
        self.phi = 0 
    
    def recovery(self,r):
        cnt=0
        # Ausgabe initialisieren 
        x_out = np.zeros([len(r)/self.n_up*2,1],complex) 
        f = 0 
        # Phasenkorrektur 
        for ii in range(0,len(r)): 
        #Korrektur des Eingangswertes 
            r[ii] = r[ii] * np.exp(1j*(-self.phi)) 
     
        #neues Phaseninkrement berechnen 
            self.phi = self.phi + 2*np.pi*self.nu/8000
            #print(self.phi)
        #Filteroperationen, MF+DMF 
            (x,self.x_buf)=sig.lfilter(self.g_mf, 1,[r[ii]],0,self.x_buf)        
            (y,self.y_buf)=sig.lfilter(self.g_dmf, 1,[r[ii]],0,self.y_buf)
            ##tested, gleich wie in MATLAB
        #Downsample by 8 
            if (np.mod(ii,self.n_up) ==0 ): 
        #Fehler berechnen 
                    e = 0.5*np.imag(self.x_1*np.conj(self.y_1)) + 0.5*np.imag(x*np.conj(y))
                    #print("e=",e)
                    self.nu= self.nu + self.gamma*e[0];      
        #Frequenzoffset berechnen 
                    f=f+self.nu; 
       
            if (np.mod(ii,self.n_up/2) == 0):     
                self.x_1 = x
                self.y_1 = y    
        #Ausgabe, Faktor 2 ueberabgetastet! 
                x_out[cnt] = x
                cnt=cnt+1
        f=f/len(r)*self.n_up
        print(f)
        return x_out,f
   #kommentar: funktioniert, braucht aber viel laengere Konvergenzzeit
   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Modified delay correlation
def DC(r,T,symbols_known,n_up,L,k):
    #k= Anzahl der Wiederholungen, L//k=Fensterlaenge 
    d=np.asarray([symbols_known[np.mod(i,L//k)]*np.conj(symbols_known[np.mod(i,L//k)+L//k]) for i in range(len(r)-n_up*L//k*2)])
    Pd = np.asarray([np.sum(np.conj(r[i:i+n_up*L//k:n_up])*r[i+L//k*n_up:i+L//k*2*n_up:n_up]*d[i]) for i in range(len(r)-n_up*L//k*2)])
    Rd = np.asarray([np.sum(np.abs(r[i+L*n_up//k:i+L//k*2*n_up:n_up])**2) for i in range(len(r) - L//k*2*n_up)])  
    M = np.abs(Pd/Rd)**2
#    plt.plot(Pd)
#    plt.plot(Rd)
    plt.figure()    
    plt.stem(M)
#    np.argmax(M)/n_up
    f_est=1/(2*np.pi*L//k*T)*np.angle(Pd[np.argmax(M)])
    plt.figure()
    plt.plot(1/(2*np.pi*L//k*T)*np.angle(Pd))
    m=-1
    #f_est=-1
    cnt=0
    if(np.count_nonzero(M>np.max(M)*0.5) > L//8*6*n_up*0.1):            
        for i in range(0,M.size-n_up*L//k*(k-2)):
            if (np.count_nonzero(M[i:i+L//k*(k-1)*n_up:k*n_up]>0.7)>=k-1 and np.count_nonzero(M[i:i+L//k*(k-2)*n_up:k*n_up]>0.9*np.max(M))>0):
                #threshold soll auf SNR angepasst werden
                m+=i/n_up               
                #f_est=1/(2*np.pi*L//k*T)*np.angle(Pd[i+4*L//k*n_up])
                cnt+=1
        if cnt==0:
            m=-1
        else:
            m=m/cnt
            f_est=1/(2*np.pi*L//k*T)*np.angle(Pd[int(m*n_up+1.5*k*n_up)])   
            print(m+1.5*k*n_up)
    return f_est,m,M
            




