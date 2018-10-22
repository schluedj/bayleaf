### Family storage for bayleaf
### Author: David Schlueter
### David Schlueter
### Vanderbilt University Department of Biostatistics,
### File contains many hacks to GLM module in pymc3
### July 21, 2017

import theano.tensor as tt
from pymc3.model import Model, Deterministic
import numpy as np
import pymc3.distributions as pm_dists
import pandas as pd
from copy import copy
from . import likelihoods
from .likelihoods import *
#from .likelihoods import ExtremeValue_Censored
from pymc3.model import modelcontext
import numbers

FLOAT_EPS = np.finfo(float).eps

class Family(object):
    """Base class for Family of likelihood distribution
    This is a hack from pymc3 GLM
    """
    priors = {}
    link = None

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

    def _get_priors(self, model=None, name=''):
        """Return prior distributions of the likelihood.
        Returns
        -------
        dict : mapping name -> pymc3 distribution
        """
        if name:
            name = '{}_'.format(name)
        model = modelcontext(model)
        priors = {}
        for key, val in self.priors.items():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var('{}{}'.format(name, key), val)

        return priors

    def create_likelihood(self, name, y_est, y_data, e_data, model=None):
        """Create likelihood distribution of observed data.
        Parameters
        ----------
        y_est : theano.tensor
            Estimate of dependent variable
        y_data : array
            Observed dependent variable
        e_data: array
            Observed censoring indicator
        """
        priors = self._get_priors(model=model, name=name)
        # Wrap y_est in link function
        priors[self.parent] = y_est
        if name:
            name = '{}_'.format(name)
        return self.likelihood('{}y'.format(name), observed={'value':y_data, 'event': e_data}, **priors)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent})
    Priors       : {priors}.""".format(klass=self.__class__, likelihood=self.likelihood.__name__, parent=self.parent, priors=self.priors)

################################################################################
###################### Univariate Parametric Models ############################
################################################################################

#class Gamma(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
#    link = tt.exp
#    likelihood = Gamma_Censored
#    parent = 'indep'#parameter that links the indep component to the
#    priors = {}

class Exponential(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    link = tt.exp
    likelihood = Exponential_Censored
    parent = 'indep'#parameter that links the indep component to the
    priors = {'alpha': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}

class Weibull(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    link = tt.exp
    likelihood = Weibull_Censored
    parent = 'indep'#parameter that links the indep component to the
    priors = {'alpha': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}

#class Extreme_Value(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
#    link = tt.exp
#    likelihood = ExtremeValue_Censored
#    parent = 'indep'#parameter that links the indep component to the
#    priors = {'alpha': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}

class Weibull_PH(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    link = tt.exp
    likelihood = WeibullPH
    parent = 'indep'#parameter that links the indep component to the
    priors = {'lam': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.),
              'alpha':pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}



#### Multivariate Models
#### Copulas
class CopulaFamily(object):
    """Base class for Family of likelihood distribution
    This is a hack from pymc3 GLM
    """
    priors = {}
    link = None

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

    def _get_priors(self, model=None, name=''):
        """Return prior distributions of the likelihood.
        Returns
        -------
        dict : mapping name -> pymc3 distribution
        """
        if name:
            name = '{}_'.format(name)
        model = modelcontext(model)
        priors = {}
        for key, val in self.priors.items():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var('{}{}'.format(name, key), val)

        return priors

    def create_likelihood(self, name, indep_1, indep_2, time_1, time_2,
                          e_1, e_2, model=None):
        """Create likelihood distribution of observed data.
        Parameters
        ----------
        """
        priors = self._get_priors(model=model, name=name)
        priors[self.parent_1] = indep_1
        priors[self.parent_2] = indep_2
        if name:
            name = '{}_'.format(name)
        return self.likelihood('{}y'.format(name), observed={"time_1":time_1, "time_2":time_2,'delta_1':e_1, 'delta_2': e_2}, **priors)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent_1},{parent_2})
    Priors       : {priors}.""".format(klass=self.__class__, likelihood=self.likelihood.__name__, parent_1 =self.parent_1, parent_2 = self.parent_2, priors=self.priors)

################################################################################
###################### Bivariate ############################
##

class Clayton_Trans(CopulaFamily):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    likelihood = Clayton_Censored_Trans
    parent_1 = 'indep_1'
    parent_2 = 'indep_2'
    #parameter that links the indep component to the
    priors = {'alpha':pm_dists.HalfCauchy.dist(beta=5), #testval=1.),
             'lam_1': pm_dists.HalfCauchy.dist(beta=2.5), #testval=.1),
             'rho_1': pm_dists.HalfCauchy.dist(beta=2.5), #testval=.1),
            'lam_2': pm_dists.HalfCauchy.dist(beta=2.5), #testval=.1),
            'rho_2': pm_dists.HalfCauchy.dist(beta=2.5),
             'r_1':pm_dists.HalfCauchy.dist(beta=2.5),
             'r_2':pm_dists.HalfCauchy.dist(beta=2.5)} #testval=.1)}
    
    
#### Frailty stuff



### Finally, we have the interplay between the Family and the Likelihoods
class FrailtyFamily(object):
    """Base class for Family of likelihood distribution
    This is a hack from pymc3 GLM
    """
    priors = {}
    link = None

    def __init__(self, **kwargs):
        # Overwrite defaults
        for key, val in kwargs.items():
            if key == 'priors':
                self.priors = copy(self.priors)
                self.priors.update(val)
            else:
                setattr(self, key, val)

    def _get_priors(self, model=None, name=''):
        """Return prior distributions of the likelihood.
        Returns
        -------
        dict : mapping name -> pymc3 distribution
        """
        if name:
            name = '{}_'.format(name)
        model = modelcontext(model)
        priors = {}
        for key, val in self.priors.items():
            if isinstance(val, numbers.Number):
                priors[key] = val
            else:
                priors[key] = model.Var('{}{}'.format(name, key), val)

        return priors

    def create_likelihood(self, name, coeffs_all, theta, rhos, lams, rs, time, event, event_change, x, total_size,k, model=None):
        """Create likelihood distribution of observed data.
        Parameters
        ----------
        """
        priors = self._get_priors(model=model, name=name)
        priors[self.parent_1] = coeffs_all
        priors[self.parent_2] = theta
        priors[self.parent_3] = rhos
        priors[self.parent_4] = lams
        priors[self.parent_5] = rs
        
        
        
        if name:
            name = '{}_'.format(name)
            
        ## new
        ### Here's where we pass the minibatch generator if we want minibatch advi 
        ## assume minibatch corresponds to minibatch size
        # if a minibatch, we need the total size
        if str(time) == 'Minibatch':
            return self.likelihood('{}y'.format(name), observed={"time":time,'delta_1': event, 'delta_2': event_change, 'x': x}, total_size = total_size, k=k, **priors)
        else:
            return self.likelihood('{}y'.format(name), observed={"time":time,'delta_1': event, 'delta_2': event_change, 'x': x},k=k, **priors)

    def __repr__(self):
        return """Family {klass}:
    Likelihood   : {likelihood}({parent_1},{parent_2},{parent_3},{parent_4},{parent_5})
    Priors       : {priors}.""".format(klass=self.__class__, likelihood=self.likelihood.__name__, parent_1 =self.parent_1,
                                       parent_2 =self.parent_2,
                                       parent_3 =self.parent_3,
                                       parent_4 =self.parent_4,
                                       parent_5 =self.parent_5,
                                       priors=self.priors)
##### Frailty family 
class GammaFrailty(FrailtyFamily):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    likelihood = Gamma_Frailty
    parent_1 = 'coeffs_all'
    parent_2 = 'theta'
    parent_3 = 'rhos'
    parent_4 = 'lams'
    parent_5 = 'rs'
    #parameter that links the indep component to the
    #priors = {'theta':pm_dists.HalfCauchy.dist(beta=5) #testval=1.),
     #        } #testval=.1)}



