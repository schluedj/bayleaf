##### Simulation Functions for bayleaf
### David Schlueter
### Vanderbilt Universty Department of Biostatistics
### Simulation functions
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import pandas as pd

from . import likelihoods
from . import families


__all__ = [
'sim_Weibull', 'sim_transform_Weibull',  'sim_weibull_frail_generalized', 'sim_simple_covs'
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

def sim_transform_Weibull(N, lam, rho, beta, rateC, maxtime, r):
    '''
    Function to simulate transformed weibull survival times with exponential censoring according to the weibull PH model
    Parameters
    ----------
    N : Number of samples to generate
    lam : scale parameter
    rho : shape parameter
    beta : effect size coefficients
    rateC : censoring rate of exponential censoring time
    maxtime : maximum study time
    r : transformation parameter
    '''
    x = np.random.binomial(n=1,p=.5,size =N)
    U = np.random.uniform(size=N)
    Tlat = ((np.exp(-np.log(U)*r)-1)/(lam*r*np.exp(x*beta)))**(1/rho) #probability integral transform
    C = np.random.exponential(scale=1/rateC, size = N)
    C[C > maxtime] = maxtime
    # follow-up times and event indicators
    time = np.min(np.asarray([Tlat,C]),axis = 0)
    status = Tlat <= C
    out = pd.DataFrame(np.array([time, status, x]).T)
    out.columns = ["time", "event", "x"]
    return(out)

def sim_weibull_frail_generalized(betas, theta, X, lam, r, rho, maxtime, cens_end, n, k, first = False):
    '''
    Function to simulate transformed weibull survival times with uniform censoring according to the weibull PH model
    Parameters
    ----------
    betas : effect sizes
    lam : scale parameters for different levels (must be dimension kx1)
    theta : parameter of gamma distribution of frailties
    rho : shape parameters for each level (kx1)
    X1 : covariates
    maxtime : maximum study time
    r : transformation parameter
    k : number of outcomes
    '''
    w = np.random.gamma(size = n, shape=theta**(-1), scale = theta)
    ## from probability integral transform
    Te = ((np.exp(-(np.log(np.random.uniform(size=(n,k)))*r)/w[:,None])-1)/(r*lam*np.exp(np.dot(X,betas.T))))**(1/rho)
    # generate censoring time, unif and truncated by tau
    Cens = 1+cens_end*np.random.uniform(size = n)
    Cens[Cens>maxtime] = maxtime
    alltimes = np.vstack((Cens,Te.T)).T
    eventType = []
    for i in range(len(w)):
        eventType.append(np.where(alltimes[i,]==np.amin(alltimes[i,]))[0][0])
    obs_t = list(np.amin(alltimes,axis = 1))
    out = pd.DataFrame(np.array([obs_t, eventType, pd.Series(X[:,[0]][:,0]),pd.Series(X[:,[1]][:,0]),w])).T
    # Clean up for the covariates
    out.columns = ["obs_t", "eventType", "sex", "age", "sim_frail"]
    return(out)

def sim_simple_covs(n):
    '''
    Function to simulate simple covariates
    ----------
    n : Number of samples to generate
    '''
    sex = np.random.binomial(n=1,p=.5,size =n)
    age = np.random.gamma(size=n,  shape = 10, scale = 1/.3)
    return(np.array([sex,age]).T)


