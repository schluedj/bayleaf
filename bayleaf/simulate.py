##### Simulation Functions for bayleaf

### Weibull Simulation with exponential censoring:

import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import pandas as pd

from . import likelihoods
from . import families


__all__ = [
'sim_Weibull'
]



def sim_Weibull(N, lam, rho, beta, rateC, maxtime):
    '''
    Function to simulate weibull survival times with exponential censoring according to the weibull PH model
    Parameters
    ----------
    N : Number of samples to generate
    lam : scale parameter
    rho : shape parameter
    beta : effect size coefficients
    rateC : censoring rate of exponential censoring time
    maxtime : maximum study time
    '''
    x = np.random.binomial(n=1,p=.5,size =N)
    U = np.random.uniform(size=N)
    Tlat = (-np.log(U)/(lam*np.exp(x*beta)))**(1/rho) #probability integral transform
    C = np.random.exponential(scale=1/rateC, size = N)
    C[C > maxtime] = maxtime
    # follow-up times and event indicators
    time = np.min(np.asarray([Tlat,C]),axis = 0)
    status = Tlat <= C
    out = pd.DataFrame(np.array([time, status, x]).T)
    out.columns = ["time", "event", "x"]
    return(out)
