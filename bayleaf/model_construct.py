
### Functions to build models storage for bayleaf
### Author: David Schlueter
### Vanderbilt University Department of Biostatistics
### Based on the GLM module in pymc3
### June 2, 2018

import theano.tensor as tt
from pymc3.model import Model, Deterministic
from pymc3.distributions import Normal, Flat, transforms, Bound
from pymc3.glm.utils import any_to_tensor_and_labels
import warnings
import inspect
import numpy as np
from scipy import special
from pymc3.glm import families
import pymc3.distributions as pm_dists
import patsy
import numbers
import pandas as pd
from copy import copy
import theano.tensor as tt
from pymc3.model import modelcontext
from pymc3 import distributions as pm_dists

FLOAT_EPS = np.finfo(float).eps
import pymc3 as pm

from .likelihoods import *
from .families import *



__all__ = [
    'IndependentComponent',
    'ParSurv',
    'CopulaIndependentComponent',
    'Copula'
]
## Now we need to hack the from formula call
class IndependentComponent(Model):
    """Creates independent component for independent variables, y_est is accessible via attribute
    Need to be able to hack this for usage with non-linear component
    Will be compatible with non-linear components as well
    Hack of pm.GLM.LinearComponent
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    e: pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars : dict - random variables instead of creating new ones
    """
    default_regressor_prior = Normal.dist(mu=0, tau=1.0E-6)
    default_intercept_prior = Flat.dist()

    def __init__(self, x, y, e, intercept=True, labels=None,
                 priors=None, vars=None, name='', model=None):
        super(IndependentComponent, self).__init__(name, model)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
        x, labels = any_to_tensor_and_labels(x, labels)

        # now we have x, shape and labels
        if intercept:
            x = tt.concatenate(
                [tt.ones((x.shape[0], 1), x.dtype), x],
                axis=1
            )
            labels = ['Intercept'] + labels
        self.x = x
        coeffs = list()
        for name in labels:
            if name == 'Intercept':
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(
                        name=name,
                        dist=priors.get(
                            name,
                            self.default_intercept_prior
                        )
                    )
                coeffs.append(v)
            else:
                if name in vars:
                    v = Deterministic(name, vars[name])
                else:
                    v = self.Var(
                        name=name,
                        dist=priors.get(
                            name,
                            priors.get(
                                'Regressor',
                                self.default_regressor_prior
                            )
                        )
                    )
                coeffs.append(v)
        self.coeffs = tt.stack(coeffs, axis=0)
        self.y_est = x.dot(self.coeffs)

class ParSurv(IndependentComponent):
    """Creates parametric survival model, y_est is accessible via attribute
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array, corresponds to the observed times vector
    e : pd.Series or np.array, corresponds to the event indicator vector
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    init : dict - test_vals for coefficients
    vars : dict - random variables instead of creating new ones
    family : bayleaf..families object
    """
    def __init__(self, x, y, e, intercept=True, labels=None,
                 priors=None, vars=None, family='weibull', name='', model=None):
        super(ParSurv, self).__init__(
            x, y, e, intercept=intercept, labels=labels,
            priors=priors, vars=vars, name=name, model=model
        )
        _families = dict(
            #weibullph = Weibull_PH,
            weibull = Weibull,
            exponential = Exponential,
            #extremevalue = Extreme_Value
        )
        if isinstance(family, str):
            family = _families[family]()
        self.y_est = family.create_likelihood(
            name='', y_est=self.y_est,
            y_data=y, e_data=e, model=self)

    @classmethod
    def from_formula(cls, formula, data, priors=None,
                     vars=None, family='weibull', name='', model=None):
        import patsy
        ##### Here's how we parse the formula ######
        # Parse the formula and split into essential components
        #### TODO: Automatic selection of multivariate family based on dimension of inputs
        outcomes= formula.split("~")[0]
        # get time variables
        time_vars = [v.strip() for v in outcomes[outcomes.find("([")+2:outcomes.find("]")].split(",")]
        #get event times
        event_raw = outcomes[outcomes.find("],")+2:]
        event_vars = [v.strip() for v in event_raw[event_raw.find("[")+1:event_raw.find("])")].split(",")]
        # Now get x, times, and events
        x = patsy.dmatrix(formula.split("~")[1].strip(), data)
        y = data[time_vars].as_matrix()
        e = data[event_vars].as_matrix()
        labels = x.design_info.column_names
        return cls(x=np.asarray(x), y=np.asarray(y)[:, 0], e=np.asarray(e)[:, 0] ,intercept=False, labels=labels,
                   priors=priors, vars=vars, family=family, name=name, model=model)

#### Copula Components

class CopulaIndependentComponent(Model):
    """Creates independent component for independent variables, y_est is accessible via attribute
    Need to be able to hack this for usage with non-linear component
    Will be compatible with non-linear components as well
    Hack of pm.GLM.LinearComponent
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : pd.DataFrame or np.ndarray
    y : pd.Series or np.array
    e: pd.Series or np.array
    intercept : bool - fit with intercept or not?
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars : dict - random variables instead of creating new ones
    """
    default_regressor_prior = Normal.dist(mu=0, tau=1/100)
    def __init__(self,
                 time_1, time_2,
                 e_1, e_2,
                 x, labels=None,
                 priors=None, vars=None, name='', model=None):
        super(CopulaIndependentComponent, self).__init__(name, model)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
        # we need 2 sets of these
        x, labels = any_to_tensor_and_labels(x, labels)
        # now we have x, shape and labels
        self.x = x
        labels_1 = [s + "_1" for s in labels]
        ###First Dimension
        coeffs_1 = list()
        for name in labels_1:
            if name in vars:
                v = Deterministic(name, vars[name])
            else:
                v = self.Var(
                    name=name,
                    dist=priors.get(
                        name,
                        priors.get(
                            'Regressor',
                            self.default_regressor_prior
                        )
                    )
                )
            coeffs_1.append(v)
        self.coeffs_1 = tt.stack(coeffs_1, axis=0)
        #### Second Dimension
        labels_2 = [s + "_2" for s in labels]
        coeffs_2 = list()
        for name in labels_2:
            if name in vars:
                v = Deterministic(name, vars[name])
            else:
                v = self.Var(
                    name=name,
                    dist=priors.get(
                        name,
                        priors.get(
                            'Regressor',
                            self.default_regressor_prior
                        )
                    )
                )
            coeffs_2.append(v)
        self.coeffs_2 = tt.stack(coeffs_2, axis=0)
        ### Return some stuff
        self.indep_1 = x.dot(self.coeffs_1)
        self.indep_2 = x.dot(self.coeffs_2)
        ### JUst testing
        #self.indep_1 = x.dot(np.array([1.3,0.03]))
        #self.indep_2 = x.dot(np.array([.8,0.03]))

        self.labels_1 = labels_1
        self.labels_2 = labels_2

class Copula(CopulaIndependentComponent):
    """
    """
    def __init__(self, time_1, time_2, e_1, e_2, x, family = "clayton",labels=None,
                 priors=None, vars=None, name='', model=None):
        super(Copula, self).__init__(time_1, time_2, e_1, e_2, x,labels=labels,
            priors=priors, vars=vars, name=name, model=model
        )

        _families = dict(
            ## This refers to a specific class of superclass Family
            clayton = Clayton,
            clayton_trans = Clayton_Trans
        )
        if isinstance(family, str):
            family = _families[family]()
        self.y_est = family.create_likelihood(name='',indep_1=self.indep_1, indep_2=self.indep_2,
                                              time_1=time_1, time_2=time_2,
                                              e_1=e_1, e_2=e_2,
                                              model=self)






parsurv = ParSurv
copula = Copula
