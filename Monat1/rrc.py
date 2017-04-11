import numpy as np
       
import scipy.linalg as lin
########################
# find impulse response of an RRC filter

########################
class rrcfilter:
    def __init__(self,K, n_up, t_symb, beta):
        self.K=K
        self.n_up=n_up
        self.t_symb=t_symb
        self.beta=beta
        ''' 
        Determines coefficients of an RRC filter 
        
        Formula out of: J. Huber, Trelliscodierung, Springer, 1992, S. 15
        At poles, values of wikipedia.de were used (without cross-checking)
        
        NOTE: Length of the IR has to be an odd number
        
        IN: length of IR, upsampling factor, symbol time, roll-off factor
        OUT: filter ceofficients
        '''
    
    def ir(self):
        assert self.K%2 != 0, "Filter length needs to be odd"
        if self.beta == 0:
            self.beta = 1e-32
    
        # init
        rrc = np.zeros(self.K)
        t_sample = self.t_symb/self.n_up
        
        if self.K%2 != 0:
            i_steps = np.arange(0,self.K)
            k_steps = np.arange(-(self.K-1)/2.0,(self.K-1)/2.0+1)    
            t_steps = k_steps*t_sample
            for i in i_steps:
                if t_steps[i] == 0:
                    rrc[i] = 1.0/np.sqrt(self.t_symb)*(1.0-self.beta+4.0*self.beta/np.pi)
                elif np.abs(t_steps[i]) == self.t_symb/4.0/self.beta:
                    rrc[i] = self.beta/np.sqrt(2.0*self.t_symb)*((1+2/np.pi)*np.sin(np.pi/4.0/self.beta)+(1.0-2.0/np.pi)*np.cos(np.pi/4.0/self.beta))
                else:
                    rrc[i] = 1.0/np.sqrt(self.t_symb)*(np.sin(np.pi*t_steps[i]/self.t_symb*(1-self.beta))+4.0*self.beta*t_steps[i]/self.t_symb*np.cos(np.pi*t_steps[i]/self.t_symb*(1+self.beta)))/(np.pi*t_steps[i]/self.t_symb*(1.0-(4.0*self.beta*t_steps[i]/self.t_symb)**2.0))
        rrc /= lin.norm(rrc) 
        return rrc
        
    
########################
# find impulse response of an RRC filter
########################
def get_rc_ir(K, n_up, t_symb, beta):
        
    ''' 
    Determines coefficients of an RC filter 
    
    Formula out of: Kammeyer, Nachrichtenuebertragung
    
    NOTE: Length of the IR has to be an odd number
    
    IN: length of IR, upsampling factor, symbol time, roll-off factor
    OUT: filter ceofficients
    '''

    assert K%2 != 0, "Filter length needs to be odd"
    if beta == 0:
        beta = 1e-32

    # init
    rc = np.zeros(K)
    t_sample = t_symb/n_up
    
    fN = 1.0/t_symb
    wN = 2.0*np.pi*fN
    
    i_steps = np.arange(0,K)
    k_steps = np.arange(-(K-1)/2.0,(K-1)/2.0+1)   
    t_steps = k_steps*t_sample
    for i in i_steps:
        if t_steps[i] == 0:
            rc[i] = 2.0*fN
        elif np.abs(t_steps[i]) == 1/(4.0*beta*fN):
            rc[i] = beta*fN*np.sin(np.pi/2.0/beta)
        else:
            rc[i] = 2.0*fN*np.sin(wN*t_steps[i])/wN/t_steps[i]*np.cos(beta*wN*t_steps[i])/(1.0-(4.0*beta*fN*t_steps[i])**2)
 
    return rc