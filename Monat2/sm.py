
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
from takt_synchro import gardner_timing_recovery ,feedforward_timing_sync
#import time 
from commpy.utilities import bitarray2dec  
import matplotlib.pyplot as plt   
#from joint_estimation import joint_estimation 
#import scipy.linalg as lin 
#import scipy.signal as sig 
import Plot 
#import commpy


#commpy.zcsequence(2,8)
#carrier Frequency
fc=1*1e9  # LTE
offset_range=15*1e-6
#print("f_max=",fc*offset_range)
SNR_dB=20
#=Eb/N0
#number of sender antennas
SA=2
#number of receiver antennas
RA=1
#data bits modulation order (BPSK)
M=4
mpsk_map=np.array([1,-1])
mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
#number of symbols per Frames
#Ns=2
##number of Frames
#Nf=100
##number of symbols
#N=Ns*Nf
##number of training symbols
N_known=32*4
N_simu=np.random.randint(N_known//2,N_known)
k=8
#k= Anzahl der Wiederholungen,
#symbol duration
T=1*1e-6
#print("f_vernachlaessigbar=",0.01/N/T)
#T=1
#Frequency offset
f_off=np.random.randint(-fc*offset_range,fc*offset_range)
#f_off=10000
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
#takt_off=0
print("takt_off=",takt_off)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
K=7
filter_=rrcfilter(K*n_up+1,n_up , 1,1)
g=filter_.ir()
#Plot.timesignal(g,"g")
#Channel matrix
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.array([[1-0.5*1j,1]])
f_est=[]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#fl=FLL(g,n_up)
gardner=gardner_timing_recovery(n_up)
for index_SA in range(0,1):
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#sender

    sender_=sender(N_simu,N_known,Ni,Nd,mpsk_map,filter_,k)
    sender_.send()
    print("n_start=",N_simu,"*n_up=",N_simu*n_up,"ende",(N_simu+N_known)*n_up)
    #training symbols(/bits) which may be shared with receiver, when a data-aided method is used
    symbols_known=sender_.symbols_known
    #symbols_known=ss
    symbols=sender_.symbols
    ibits=sender_.ibits
    #dbits=sender_.dbits
    ibits_known=sender_.ibits_known
    index=bitarray2dec(ibits_known)
    #dbits_known=sender_.dbits_known
   
    s_BB=sender_.bbsignal()
    group_delay = (g.size - 1) // 2
    #s_BB=s_BB[group_delay:-group_delay]
    
    #spec=sender_.anti_image(s_BB)
    

    #Plot.konstellation(s_BB,'Signal from sender')
    ###Kommentar: Nullpunkt wegen Filterdelay
    #Plot.timesignal(s_BB,"Baseband signal")
    #Plot.spectrum(s_BB,"Baseband spectrum")
    
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#channel

    receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map,s_BB)  
    r=receiver_.r
    rr=receiver_.r
    
    
    #Plot.timesignal(rr,"nach Kanal")
    #Plot.konstellation(r,'Signal after channel')
    #Plot.timesignal(r[:,0],'Signal after channel')
    #Plot.spectrum(r,'Signal after channel')
    #yi,yd=receiver_.detector(r_mf,H)
    #BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#with offsets

    off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/n_up)
    r=receiver_.r*np.repeat(off,RA).reshape([-1,RA])
    r=r[takt_off:]
    r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
    r_mf=r_mf[group_delay:-group_delay]*np.exp(1j*phi_off)


    #Plot.timesignal(r_mf[:,0],"nach MF")
    #Plot.timesignal(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.spectrum(r_mf[:,0],"nach MF")
    #Plot.spectrum(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.timesignal(receiver_.r_down[:,0],"downsampling")
    

    
    # %%%%%%%%%%%%%%%%%%%%%%
    #f_NDA=NDA(r_mf,M,T,H,n_up)
    #print("NDA:",f_NDA)
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#FLL
    #CSI=true
    
    #g_mf=g
    #
    #r=r_off_ft[:,0]
    #
    ##for i in range(0,500):
    ##    r=r*np.exp(-1j*2*np.pi*f*np.arange(sender_.bbsignal().size)*T/filter_.n_up)    
    #    #print("phi=",fl.phi)
    #x_out,f=fl.recovery(r)
    #f_est.append(f)


    #Plot.konstellation(r_mf,'Signal after FLL')
    #Plot.timesignal(r_mf,'Signal after FLL')
    #Plot.spectrum(r_mf,'Signal after FLL')

#%%%%%%%%%%%%%%%%%
#Modified Delay Correlation

    f_est,m=DC(r_mf[:,0],T,symbols_known,n_up,N_known,k)
#    f_estt,mm=DC(r_mf[m*n_up+N_known*n_up:,0],T,symbols_known,n_up,N_known,k)
    
    if f_est==0 and m==-1:
        print("Fehler bei Schätzung")
        break
    
    print("f_est1=",f_est)
    print("n_est1=",m)
#    
#    print("f_est2=",f_estt)
#    print("n_est2=",mm)
#    print("n_est-n_off=",m-N_simu)

    #f_est=1/(2*np.pi*L//2*T)*np.angle(Pd[np.argmax(M)])
    
    #f_est=f_off
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Frequency synchronisation

    off_sync=np.exp(-1j*2*np.pi*f_est*(np.arange(r_mf.shape[0])+group_delay)*T/n_up)
    y=r_mf[:,0]*off_sync.reshape(r_mf[:,0].shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Takt synchorinisation
#Feedforward

    #takt_est=feedforward_timing_sync(n_up,y,m,N_known,symbols_known)
    takt=np.zeros(n_up)
    abw=3
    y_s=y[n_up*(int(m)-abw-1):n_up*(int(m)+N_known+abw)]
    #cnt=np.zeros(2*abw+1)
    ab=np.zeros(abw*2+1)
    for i in range(0,y_s.size-2):
        diff_1r=np.real(y_s[i+1]-y_s[i])
        diff_2r=np.real(y_s[i+2]-y_s[i])
        diff_1i=np.imag(y_s[i+1]-y_s[i])
        diff_2i=np.imag(y_s[i+2]-y_s[i])
        if(np.abs(diff_1r)>np.abs(diff_2r)) and (np.abs(diff_1i)>np.abs(diff_2i)):
           takt[np.mod(i,n_up)]+=1
#           if (diff_1r-diff_2r)*y_s[i].real>0:

    takt_est=n_up-1-np.argmax(takt)
    print("takt_est=",takt_est)
    y_symbol=y_s[np.argmax(takt)::n_up]

        
        #%%%%%%%%%%%%%%%%%
#Gardner

    
    #gardner.run(y)
    #r_sync=gardner.output_symbols
    #plt.figure()
    #plt.plot(gardner.e); plt.title("Error signal e"); plt.show();
    #plt.figure()
    #plt.plot([np.rint(tau) for tau in gardner.tau]); plt.title("Estimated timing offset tau"); plt.ylim([-gardner.n_up//2-1, gardner.n_up//2+1]); plt.show();
    ##plt.stem(r[::gardner.n_up]); plt.title("Unsynchronized receive symbols"); plt.show();
    ##plt.stem(r_sync); plt.title("Synchronized receive symbols"); plt.show();    
    


        #%%%%%%%%%%%%%%%%%
        #Channel estimation
#%%%%%%%%%%%%%%%%%
#Wiener Lee

    #H_est=np.zeros([RA,SA],complex)
#    H_est=np.zeros((abw*2+1,1),complex)
#    i=np.zeros(SA)
#    for a in range(0,abw*2+1):
#        for k in range(0,N_known):
#            H_est[a,0]=np.average( y_symbol[k-abw+a]*np.conj(symbols_known[k]))
#    H_est=H_est*1/np.dot(symbols_known,np.conj(symbols_known))*symbols_known.size
#    
#%%%%%%%%%%%%%%%%%
#LMS(Rekursiv)
    
    mu=0.01
    H_est=np.zeros((abw*2+1,SA),complex)
#    plt.figure()
    for index_SA in range(0,1):
        for a in range(0,abw*2+1):
            H_est[a,index_SA]=y_symbol[a]*np.conj(symbols_known[0+N_known*index_SA])
            for k in range(1,N_known):
                e=y_symbol[k+a+N_known*index_SA]-H_est[a,index_SA]*symbols_known[k+N_known*index_SA]
                H_est[a,index_SA]+=mu*e
        print("H_est=",H_est[abw+(int(m)-N_simu),index_SA])


#%%%%%%%%%%%%%%%%%       
       
       
       
       
       #%%%%%%%%%%%%%%%%%
#    print("H_est=",H_est[abw+(int(m)-N_simu),0])
    print("H=",H[0,0],H[0,1])
            
    
    
            #%%%%%%%%%%%%%%%%%
#feine Zeitsynchro

    #cnt=[]
    #for i in range(0,y_symbol.size):
    #    if y_symbol[i]>0:
    #        y_symbol[i]=1
    #    else:
    #        y_symbol[i]=-1
    #for a in range(0,2*abw+1):
    #    cnt.append(np.sum(np.sign(y_symbol[a:a+symbols_known.size])*np.sign(symbols_known)))
    #

            #%%%%%%%%%%%%%%%%%
#reste Frequenzoffset
            
            
            
   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Detection and Test 
    #H_est=H  #with unknown CSI
    #yi,yd=receiver_.detector(r_sync,H_est)
    ##yi,yd=rr.detector(H_est,SNR_dB,mpsk_map,r_ft_syc)
    #BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits,dbits)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
#print("f_est=",f_est,", n_est=",n_est," , H_diff_max=", np.max(H-H_est))  
#print("BER for index bits=",BERi,", BER for data bits=",BERd)  
###

