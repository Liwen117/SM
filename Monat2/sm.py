
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""
#
#f_sync
#threshold nicht nur nach oben sonder auch nach unten!
import numpy as np   
from rrc import rrcfilter 
from sender import sender 
from receiver import receiver 
#import test 
from f_sync import DC 
from takt_synchro import gardner_timing_recovery 
#import time 
from commpy.utilities import bitarray2dec  
import matplotlib.pyplot as plt   
#from joint_estimation import joint_estimation 
#import scipy.linalg as lin 
#import scipy.signal as sig 
#import Plot 
#import commpy


#commpy.zcsequence(2,8)
#carrier Frequency
fc=1*1e9  # LTE
offset_range=15*1e-6
print("f_max=",fc*offset_range)
SNR_dB=10
#=Eb/N0
#number of sender antennas
SA=2
#number of receiver antennas
RA=1
#data bits modulation order (BPSK)
M=2
mpsk_map=np.array([1,-1])
#mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
#mpsk_map =1/np.sqrt(2) * np.array([1, 1j, -1j, -1], dtype=complex)
#number of symbols per Frames
Ns=2
#number of Frames
Nf=100
#number of symbols
N=Ns*Nf
#number of training symbols
N_known=64*4
N=N_known*4
k=8
#symbol duration
T=1*1e-6
#print("f_vernachlaessigbar=",0.01/N/T)
#T=1
#Frequency offset
f_off=np.random.randint(-fc*offset_range,fc*offset_range)
#f_off=np.random.randint(-0.01/T,0.01/T)
#N_known=int(1//T//f_off/4)
#N=10*N_known
print("f_off=",f_off)
#symbol offset 
#n_off=2
#phase offset
phi_off=np.random.random()*2*np.pi
phi_off=0
#number of Index bits per symbol
Ni=int(np.log2(SA))
#number of Data bits per symbol
Nd=int(np.log2(M))
#Upsampling rate
n_up=8
takt_off=np.random.randint(1,n_up)
print("takt_off=",takt_off)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
K=40
filter_=rrcfilter(K*n_up+1,n_up , 1,1)
g=filter_.ir()
#Plot.spectrum(g,"g")
#Channel matrix
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.array([[0.5,0.1]])
#H=np.ones([RA,SA])
f_est=[]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#fl=FLL(g,n_up)
gardner=gardner_timing_recovery(n_up)
for i in range(0,1):
    #sender
    sender_=sender(N,N_known,Ni,Nd,mpsk_map,filter_,k)
    print("n_start=",sender_.n_start,sender_.n_start*n_up,(sender_.n_start+N_known)*n_up)
    #training symbols(/bits) which may be shared with receiver, when a data-aided method is used
    symbols_known=sender_.symbols_known
    #symbols_known=ss
    symbols=sender_.symbols
    ibits=sender_.ibits
#    dbits=sender_.dbits
    ibits_known=sender_.ibits_known
    index=bitarray2dec(ibits_known)
   # dbits_known=sender_.dbits_known
    
    
    s_BB=sender_.bbsignal()
    group_delay = (g.size - 1) // 2
#    s_BB=s_BB[group_delay:-group_delay]
    
    #spec=sender_.anti_image(s_BB)
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(s_BB,'Signal from sender')
    ###Kommentar: Nullpunkt wegen Filterdelay
    #Plot.timesignal(s_BB,"Baseband signal")
    #Plot.spectrum(s_BB,"Baseband spectrum")
    
    #!!Image filterung
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #with Filter
    receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map)
    
    r=receiver_.r
    rr=receiver_.r
    #Plot.timesignal(rr,"nach Kanal")
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(r,'Signal after channel')
    #Plot.timesignal(r[:,0],'Signal after channel')
    #Plot.spectrum(r,'Signal after channel')
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
#    off=np.exp(1j*2*np.pi*f_off*np.arange(r.shape[0])*T/filter_.n_up)
#    r_off_ft=r*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)
#    
#    
    
    #sp=np.fft.fft(r)
    #yi,yd=receiver_.detector(r_mf,H)
    #BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
    
    
    
    #with offsets
    #Frequency offset before MF(+filter length) 
    #!!!T anpassen!!!
    off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/n_up)
    r=receiver_.r*np.repeat(off,RA).reshape([-1,RA])
    r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
    r_mf=r_mf[2*group_delay:-2*group_delay]


    #Plot.timesignal(r_mf[:,0],"nach MF")
    #Plot.timesignal(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.spectrum(r_mf[:,0],"nach MF")
    #Plot.spectrum(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.timesignal(receiver_.r_down[:,0],"downsampling")
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(r_mf,'Signal after MF')
    #Plot.timesignal(r_mf,'Signal after MF')
    #Plot.spectrum(r_mf,'Signal after MF')
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #offset after MF
    #r=receiver_.channel()
    #r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
    #off=np.exp(1j*2*np.pi*f_off*np.arange(r_mf.shape[0])*T/filter_.n_up)
    #
    ##
    #r_off_ft=r_mf*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)
    #r_off_ft=np.concatenate((r_off_f[n_off:],r_off_f[:n_off]))*np.exp(1j*2*np.pi*phi_off)
    
    # %%%%%%%%%%%%%%%%%%%%%%
    #f_NDA=NDA(r_mf,M,T,H,n_up)
    #print("NDA:",f_NDA)
    
    
    
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #FLL
    #CSI=true
    
#    g_mf=g
#    
#    r=r_off_ft[:,0]
#    
#    #for i in range(0,500):
#    #    r=r*np.exp(-1j*2*np.pi*f*np.arange(sender_.bbsignal().size)*T/filter_.n_up)    
#        #print("phi=",fl.phi)
#    x_out,f=fl.recovery(r)
#    f_est.append(f)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#print(DC(r_mf[:,0],T,dbits_known,M))
#y=r_mf[:,0]
#D=8
#C=np.zeros(y.size,complex)
#P=np.zeros(y.size,complex)
#M=np.zeros(y.size,complex)
#d=np.zeros(y.size+D,complex)
#for n in range(D+1,y.size-D):
#    for i in range(0,n-D):
#            d[i]=symbols[i//n_up]*np.conj(symbols[(i+D)//n_up])
#            P[n]+=np.abs(y[n-i-D])**2
#            C[n]+=np.conj(y[n-i])*y[n-i-D]*d[i]
#            M[n]=np.abs(C[n])/P[n]
#m=np.argmax(M)
#f_est=1/(2*np.pi*D*T)*np.angle(C[m])
#print(f_est,m)


#%%%%%%%%%%%%%%%%%
#Modified Delay Correlation
    f_est,m,M=DC(r_mf,T,symbols_known,n_up,N_known,k)
#    if m==-1:
#        break
        #Schätzung fehlerbehaftet, kein ACK fuer Sender, Kanalschätzung wird nicht eingeschalten. 
        #das gleiche Frame wird noch mal gesendet werden.
#    r=r_mf
#    L=N_known
#    Pd = np.asarray([np.sum(np.conj(r[i:i+n_up*L//2:n_up])*r[i+L//2*n_up:i+L*n_up:n_up]*symbols_known[np.mod(i,L//2)]*np.conj(symbols_known[np.mod(i,L//2)+L//2]) ) for i in range(len(r)-n_up*L)])
#    Rd = np.asarray([np.sum(np.abs(r[i+L*n_up//2:i+L*n_up:n_up])**2) for i in range(len(r) - L*n_up)])  
#    M = np.abs(Pd/Rd)**2
#    m=np.argmax(M)/n_up
    print("n_est=",m)
#    #plt.plot(M)
#    f_est=1/(2*np.pi*L//2*T)*np.angle(Pd[np.argmax(M)])
#    
    print("f_est=",f_est)
#    plt.plot(np.abs(M)); plt.ylabel("M"); plt.xlabel("d [Samples]"); plt.xlim([0, len(M)]); plt.ylim([0,1.1]); plt.title("M(d)"); plt.show()
#    plt.plot(np.angle(Pd)); plt.title("arg(P(d))"); plt.xlabel("Verschiebung"); plt.xlim([0, len(M)]); plt.show()
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Frequency synchronisation
    yy=r_mf*np.exp(-1j*2*np.pi*f_off*(np.arange(r_mf.shape[0])+2*group_delay)*T/n_up).reshape([-1,RA])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Takt synchorinisation
    takt=np.zeros(n_up)
    y=yy[takt_off:]
    for i in range(0,y.size-2):
        if(np.abs( np.real(y[i+2]-y[i]))< np.abs(np.real(y[i+1]-y[i]))):
           takt[np.mod(i,n_up)]+=1
    print("takt_est=",n_up-1-np.argmax(takt))
    ##Kommentar: braucht kein Vorwissen aber laengere Beobachtung
    #ABER! Nach der Preamble fuer Sync kommt es Preamble zur Kanalschaetzung
    #??????
    

        

    
        
        
        #%%%%%%%%%%%%%%%%%
#    #Gardner
#    y=r_mf[5:,0]
#    #y=x_out[3:,0]
#    gardner.run(y[:,0])
#    r_sync=gardner.output_symbols
#    plt.plot(gardner.e); plt.title("Error signal e"); plt.show();
#    plt.plot([np.rint(tau) for tau in gardner.tau]); plt.title("Estimated timing offset tau"); plt.ylim([-gardner.n_up//2-1, gardner.n_up//2+1]); plt.show();
#    #plt.stem(r[::gardner.n_up]); plt.title("Unsynchronized receive symbols"); plt.show();
#    #plt.stem(r_sync); plt.title("Synchronized receive symbols"); plt.show();    
#        

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot.konstellation(r_mf,'Signal after FLL')
#Plot.timesignal(r_mf,'Signal after FLL')
#Plot.spectrum(r_mf,'Signal after FLL')


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##### Frequency estimation without n_offset
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
###Frequency offset estimation with ML-Approximation(data-aided) with known channel for each antenna
#f_est=ML_approx_known(r_off_ft[sender_.n_start:sender_.n_start+N_known],T,symbols_known,ibits_known,H)[0]
#H_est=H
#n_est=sender_.n_start
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###ML with channel unknown
###f_est1=ML_unknown(r_off_f,T,symbols,ibits)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###ML_approx with channel unknown
##f_est1=ML_approx_unknown(r_off_ft[sender_.n_start:sender_.n_start+N_known],T,symbols_known,ibits_known)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
####Channel Estimation
###H_est=np.zeros([RA,SA],complex)  
###index=bitarray2dec(ibits)
###i=np.zeros(SA)
###for k in range(0,symbols.size):
###    H_est[:,index[k]]+= r[k,:]/symbols[k]*np.exp(-1j*2*np.pi*T*f_est*k)
###    i[index[k]]=i[index[k]]+1
####Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden
###H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
###H_diff=H-H_est
##
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###speed test
###t1=time.clock()
###t=time.clock()-t1
##
#
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##coarse Estimation for f_off
#f_off_coarse=0
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##coarse synchronisation for f_off
#off_syc=np.exp(-1j*2*np.pi*f_off_coarse*np.arange(r_mf.shape[0])*T/filter_.n_up)
#r_syc_coarse=r_off_ft*np.repeat(off_syc,RA).reshape([-1,RA])
#print(f_off-f_off_coarse)
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Joint Estimation for f_off,n_start and CSI 
##j=joint_estimation()
##j.function(r_syc_coarse,N,N_known,T,ibits_known,symbols_known,SA,RA)
##f_est=j.f_est
##n_est=j.n_est
##H_est=j.H_est
   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Frequency synchronisation
#off_syc=np.exp(-1j*2*np.pi*f_est*(np.arange(r_mf.shape[0]))*T)
#r_f_syc=r_syc_coarse*np.repeat(off_syc,RA).reshape([-1,RA])

   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Test for the Joint estimation
#if CSI==true:
#    H_est=H  #with unknown CSI
#yi,yd=receiver_.detector(r_f_syc,H_est)
##yi,yd=rr.detector(H_est,SNR_dB,mpsk_map,r_ft_syc)
#BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits,dbits)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
#print("f_est=",f_est,", n_est=",n_est," , H_diff_max=", np.max(H-H_est))  
#print("BER for index bits=",BERi,", BER for data bits=",BERd)  
###

