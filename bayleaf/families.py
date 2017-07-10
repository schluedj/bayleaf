### Family storage for bayleaf
### Author: David Schlueter
### Vanderbilt University Department of Biostatistics,
### File contains many hacks to GLM module in pymc3
### July 10, 2017

import theano.tensor as tt
from pymc3.model import Model, Deterministic
import numpy as np
import pymc3.distributions as pm_dists
import pandas as pd
from copy import copy
from pymc3.model import modelcontext
FLOAT_EPS = np.finfo(float).eps
from . import likelihoods

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

class Extreme_Value(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    link = tt.exp
    likelihood = ExtremeValue_Censored
    parent = 'indep'#parameter that links the indep component to the
    priors = {'alpha': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}

class Weibull_PH(Family):
    # Weibull survival likelihood, accounting for censoring
    ## need to define this as likelihood
    link = tt.exp
    likelihood = WeibullPH
    parent = 'indep'#parameter that links the indep component to the
    priors = {'lam': pm_dists.HalfCauchy.dist(beta=2.5, testval=1.),
              'alpha':pm_dists.HalfCauchy.dist(beta=2.5, testval=1.)}
