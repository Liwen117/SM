#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
from commpy.utilities import bitarray2dec 

#data-aided ML Approximation (Voraussetzung:f_off<<1/T)
#zuverbessern!!! jetzt wird nur von f:0~1Hz geschaetzt
def ML_approx(filter_,r,T,symbols,ibits,H):
    f_delta=100
    group_delay = (filter_.ir().size - 1) // 2
    p=np.zeros([f_delta,H.shape[0]],complex)
    f_o=np.zeros(H.shape[0])
    r_up=np.zeros([r.shape[0]-filter_.ir().size+1,H.shape[0]],complex)
    r_=np.zeros([symbols.size,H.shape[0]])
    for j in range(0,H.shape[0]):
        a= np.convolve(filter_.ir(), r[:,j])
        r_up[:,j]= a[ 2*group_delay: - 2*group_delay]
        r_[:,j] = r_up[::filter_.n_up,j]
        s_a_index=bitarray2dec(ibits)
        for f in range(0,f_delta):
            for i in range(0,r_.shape[0]):
                off_=np.exp(-1j*2*np.pi*f/f_delta*T*i)
                p[f,j]+=symbols[i]*r_[i,j]*H[j,s_a_index[i]]*off_
        f_o[j]=np.argmax(np.abs(p[:,j]))/f_delta
    print(f_o)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#non data-aided
def FLL(r,g_mf):
    k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1)
    T=1
    g_dmf= 2*np.pi*T*k[:]*g_mf[:]
    # Ausgabe initialisieren
    x_out = np.zeros([len(r)/4,1],complex)
    f = 0
    #
    #Schrittweite
    gamma = 0.01
    
    # temp. Variable: Speicherinhalt x-Zweig
    #persistent x_buf
    #if isempty(x_buf)
    x_buf = np.zeros([len(g_mf)-1,1],complex);
    #end
    #
    #persistent x_1
    #if isempty(x_1)
    x_1 = complex(0);
    #end
    #
    #% temp. Variable: Speicherinhalt y-Zweig
    #persistent y_buf
    #if isempty(y_buf)
    y_buf = np.zeros([len(g_dmf)-1,1],complex);
    #end
    #
    #persistent y_1
    #if isempty(y_1)
    y_1 = complex(0);
    #end
    #
    #% Frequenz
    #persistent nu
    #if isempty(nu)
    nu = 0;
    #end
    #
    #% Phaseninkrement
    #persistent phi
    #if isempty(phi)
    phi = 0;
    #end
    #
    #% temp. Zaehlervariable
    cnt=1;
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Phasenkorrektur
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    for ii in range(0,len(r)+1):
    #    
    #    % Korrektur des Eingangswertes
        r[ii] = r[ii] * np.exp(1j*(-phi))
    #    
    #    % neues Phaseninkrement berechnen
        phi = phi + 2*np.pi*nu/8000
    #    
    #    % Filteroperationen, MF+DMF
        x=np.convolve(g_mf,r[ii])[:r[ii].size]
        y=np.convolve(g_dmf,r[ii])[:r[ii].size]
        
    #    
    #      
    #    % Downsample by 8
        if (np.mod(ii,8) == 1):
    #        
    #        % Fehler berechnen
            e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y)); 
    #        
    #        % Frequenzoffset berechnen
            nu= nu + gamma*e;     
    #        
    #        % nur zur Visualisierung
            f=f+nu;
    #    end
    #    
    #    % Downsample by 4
        if (np.mod(ii,4) == 1):
    #        
            x_1 = x;
            y_1 = y;
    #        
    #        % Ausgabe, Faktor 2 ueberabgetastet!
            x_out[cnt] = x;
            cnt=cnt+1;
    #    end    
    #   
    #end
    #
    #% nur zur Visualisierung
    print(f/len(r)*8)
    return x_out
