### Functions to build models storage for bayleaf
### Author: David Schlueter
### Vanderbilt University Department of Biostatistics
### Based on the GLM module in pymc3
### July 12, 2018

import theano.tensor as tt
from pymc3.model import Model, Deterministic
from pymc3.distributions import Normal, Flat, transforms, Bound, Gamma, InverseGamma
from pymc3.glm.utils import any_to_tensor_and_labels
from pymc3 import find_MAP
import warnings
import inspect
import numpy as np
import theano
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
    'CopulaIndependentComponent'
    'FrailtyIndependentComponent',
    'Copula',
    'Frailty'
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
        ### create independent components as attributes of self.
        self.indep_1 = x.dot(self.coeffs_1)
        self.indep_2 = x.dot(self.coeffs_2)
        self.labels_1 = labels_1
        self.labels_2 = labels_2

class Copula(CopulaIndependentComponent):
    """
    This is a hack of the glm module in pymc3.
    """
    def __init__(self, time_1, time_2, e_1, e_2, x, family = "clayton",labels=None,
                 priors=None, vars=None, name='', model=None):
        super(Copula, self).__init__(time_1, time_2, e_1, e_2, x,labels=labels,
            priors=priors, vars=vars, name=name, model=model
        )

        _families = dict(
            ## This refers to a specific class of superclass Family
            clayton_trans = Clayton_Trans
        )
        if isinstance(family, str):
            family = _families[family]()
        self.y_est = family.create_likelihood(name='',indep_1=self.indep_1, indep_2=self.indep_2,
                                              time_1=time_1, time_2=time_2,
                                              e_1=e_1, e_2=e_2,
                                              model=self)
        
        # Now for the classmethod for the formula
    @classmethod
    def from_formula(cls, formula, data, priors=None,
                         vars=None, name='',family ="clayton_trans", model=None):
        import patsy        
        outcomes= formula.split("~")[0]
            # get time variables
        time_vars = [v.strip() for v in outcomes[outcomes.find("([")+2:outcomes.find("]")].split(",")]
        #get event times
        event_raw = outcomes[outcomes.find("],")+2:]
        event_vars = [v.strip() for v in event_raw[event_raw.find("[")+1:event_raw.find("])")].split(",")]
        # Now get x, times, and events
        x = patsy.dmatrix(formula.split("~")[1].strip(), data)
        times = data[time_vars].as_matrix()
        events = data[event_vars].as_matrix()
        labels = x.design_info.column_names

        time_1 = times[:,0]
        time_2 = times[:,1]
        event_1 = events[:,0]
        event_2 = events[:,1]
        # now convert to tensors 
        x_tensor = theano.shared(np.asarray(x)+0., borrow = True)
        time1_tensor = theano.shared(time_1+0., borrow = True)
        time2_tensor = theano.shared(time_2+0., borrow = True)
        event1_tensor = theano.shared(event_1+0., borrow = True)
        event2_tensor = theano.shared(event_2+0., borrow = True)


        return cls(x=x_tensor, time_1=time1_tensor, time_2=time2_tensor, e_1=event1_tensor, e_2=event2_tensor, family = family, labels=labels,priors=priors, vars=vars, name=name, model=model)

        

###### Frailty model as of 10/22/2018

# First need a frailty constructor
class FrailtyIndependentComponent(Model):
    """
    General Class for transformation multivariate frailty model
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
    # First thing we need to do is define default priors for the model parameters
    default_regressor_prior = Normal.dist(mu=0, tau=1/100)
    default_lambda_prior = Gamma.dist(0.001,0.001, testval = 1.)
    default_rho_prior = Gamma.dist(0.001,0.001,  testval = 1.)
    default_r_prior = InverseGamma.dist(alpha =1.,  testval = 1.)
    default_theta_prior = Gamma.dist(0.001,0.001,  testval = 1.)
    def __init__(self, 
                 time,
                 event,
                 x, minibatch = 1 ,labels=None,
                 priors=None, vars=None, name='', model=None):
        
        super(FrailtyIndependentComponent, self).__init__(name, model)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
            
        ### first thing to do is determine whether we are working with tensors or np.matrices
        
        # if we are working with a matrix, we need to grab the value of the array that populates it 

        if str(time) == '<TensorType(float64, matrix)>':             
            data_tensor = True
            self.k = k  = time.get_value().shape[1] # outcome dimentionality
            self.n = n = time.get_value().shape[0] # total number of observations
            self.p = p = x.get_value().shape[1] # number of covariates
            
        else:
            
            self.k = k = time.shape[1] # outcome dimentionality
            self.n = n = time.shape[0] # total number of observations
            self.p = p = x.shape[1] # number of covariates
            
        x, labels = any_to_tensor_and_labels(x, labels) # might need to do this for the other variables 
        
        ## now for secondary delta for the gamma frac
        if data_tensor == True:
            
            # Create tensor variable for the gamma_frac component of the likelihood
            
            self.event_change = event_change = theano.shared(np.array([np.append(np.repeat(1, s), np.repeat(0, k-s)).tolist()\
                                                                       for s in np.sum(event.get_value(), axis = 1)]), borrow = True)
        else:
            self.event_change = event_change = np.array([np.append(np.repeat(1, s), np.repeat(0, k-s)).tolist()\
                                                         for s in np.sum(event, axis = 1)])
            
        ## Keep track of total size of the dataset, for minibatching 
        ## new 10.10.2018
        # If minibatch, then we need the x component to be a generator and not just a tensor 
        # by this step in the computation, X is already in tensor form 
        
        if minibatch >= 2: # kinda hacky but whatever, we can fix this later 
            # If we're using mini-batch, then we have to tell the inner workings to fix the MAP estimate 
            self.use_ADVI = True
            minibatch = int(minibatch) #just in case some n00b puts in a double/float here 
            x_mini = pm.Minibatch(data = x.get_value(), batch_size = minibatch) # make minibatch instance of the design matrix
            time_mini = pm.Minibatch(data = time.get_value(), batch_size = minibatch) # minibatch instance of the time array
            event_mini = pm.Minibatch(data = event.get_value(), batch_size = minibatch) # minibatch instance of the event array
            event_change_mini = pm.Minibatch(data = event_change.get_value(), batch_size = minibatch) # minibatch instance of the transformed event array
            
            ## assign self. attributes to later parameterize the logp function 
            
            self.x = x_mini
            self.time = time_mini
            self.event = event_mini
            self.event_change = event_change_mini
            
        else:
            # if not minibatching, just pass the tensors as they are
            self.x = x
            self.time = time
            self.event = event
            self.event_change = event_change
            
        # now we have x, shape and labels
        
        # init a list to store all of the parameters that go into our likelihood
        coeffs_all = list()
        lams = list()
        rhos = list()
        rs = list()
        
        for level in range(k): # for each dimension, instantiate a covariate effect for each predictor
            
            labels_this = [s + "_"+str(level) for s in labels]

            coeffs_this = list()
            for name in labels_this:
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
                coeffs_this.append(v)
            coeffs_this = tt.stack(coeffs_this, axis=0)
            coeffs_all.append(coeffs_this)
            
            ### Now for the baseline hazard portions 

            lam_name = 'lam_'+str(level)
            lam = self.Var(
                name = lam_name,
                dist = priors.get(lam_name,
                                  priors.get('lam', self.default_lambda_prior)
                                 )
            )# create labels for the lambdas
            lams.append(lam)
            # rhos
            rho_name = 'rho_'+str(level)
            rho = self.Var(
                name = rho_name,
                dist = priors.get(rho_name,
                                  priors.get('rho', self.default_rho_prior)
                                 )
            )
            rhos.append(rho)
            
            # finally, transformation parameters r
            r_name = 'r_'+str(level)
            r = self.Var(
                name = r_name,
                dist = priors.get(r_name,
                                  priors.get('r', self.default_r_prior)
                                 )
            )
            rs.append(r)
            
        # frailty parameter 
        theta = self.Var(
                name = 'theta',
                dist = priors.get('theta',
                                  priors.get('Theta', self.default_theta_prior)
                                 ))
        # make self attribute for the coefficients 
        self.coeffs_all = coeffs_all
        
        # changing 10.18 
        self.theta = theta
        self.lams = lams = tt.stack(lams, axis = 0)
        self.rhos = rhos = tt.stack(rhos, axis = 0)
        self.rs = rs = tt.stack(rs, axis = 0)
        
class Frailty(FrailtyIndependentComponent):
    """
    General class for inference of transformation frailty model. Inherits from FrailtIndependentComponent.
    
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : tensor, covariate informatio (n x p dimension), if minibatched, then this is a generator
    time : tensor, time variable of (n x k dimension), if minibatched, then this is a generator
    event: tensor, event indicator variable of (n x k dimension), if minibatched, then this is a generator
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars : dict - random variables instead of creating new ones
    """
    def __init__(self, time, event, x, minibatch ='', family = 'gamma' ,labels=None,
                 priors=None, vars=None, name='', model=None):
        super(Frailty, self).__init__(time, event, x, minibatch, labels=labels,
            priors=priors, vars=vars, name=name, model=model
        )
        
        _families = dict(

            gamma = GammaFrailty
            
        )
        if isinstance(family, str):
            family = _families[family]()
        
        self.y_est = family.create_likelihood(name='', coeffs_all=self.coeffs_all, 
                                              rhos = self.rhos, 
                                              lams = self.lams, 
                                              rs =self.rs,
                                              theta = self.theta,
                                              time=self.time, event = self.event, event_change = self.event_change, x = self.x,
                                              total_size = self.n,
                                              k = self.k,
                                              model=self)
        
        
        
        
    @classmethod
    def from_formula(cls, formula, data, minibatch = False, priors=None,
                     vars=None, name='', model=None):
        import patsy        
        outcomes= formula.split("~")[0]
        # get time variables
        time_vars = [v.strip() for v in outcomes[outcomes.find("([")+2:outcomes.find("]")].split(",")]
        #get event times
        event_raw = outcomes[outcomes.find("],")+2:]
        event_vars = [v.strip() for v in event_raw[event_raw.find("[")+1:event_raw.find("])")].split(",")]
        # Now get x, times, and events
        x = patsy.dmatrix(formula.split("~")[1].strip(), data)
        time = data[time_vars].as_matrix()
        event = data[event_vars].as_matrix()
        labels = x.design_info.column_names
        # add the data tensors to the computational graph
        x_tensor = theano.shared(np.asarray(x)+0., borrow = True)
        time_tensor = theano.shared(time+0., borrow = True)
        event_tensor = theano.shared(event+0., borrow = True)
  
        return cls(x=x_tensor, time=time_tensor, event=event_tensor, minibatch=minibatch, labels=labels,
                    priors=priors, vars=vars, name=name, model=model)

## Make an independent component for the fixed version, there

class FrailtyIndependentComponent_Fix(Model):
    """
    General Class for transformation multivariate frailty model
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
    # First thing we need to do is define default priors for the model parameters
    default_regressor_prior = Normal.dist(mu=0, tau=1/100)
    default_lambda_prior = Gamma.dist(0.001,0.001, testval = 1.)
    default_rho_prior = Gamma.dist(0.001,0.001,  testval = 1.)
    default_theta_prior = Gamma.dist(0.001,0.001,  testval = 1.)
    def __init__(self, 
                 time,
                 event, rs,
                 x, minibatch = 1 ,labels=None,
                 priors=None, vars=None, name='', model=None):
        
        super(FrailtyIndependentComponent_Fix, self).__init__(name, model)
        if priors is None:
            priors = {}
        if vars is None:
            vars = {}
            
        ### first thing to do is determine whether we are working with tensors or np.matrices
        
        # if we are working with a matrix, we need to grab the value of the array that populates it 

        if str(time) == '<TensorType(float64, matrix)>':             
            data_tensor = True
            self.k = k  = time.get_value().shape[1] # outcome dimentionality
            self.n = n = time.get_value().shape[0] # total number of observations
            self.p = p = x.get_value().shape[1] # number of covariates
            
        else:
            
            self.k = k = time.shape[1] # outcome dimentionality
            self.n = n = time.shape[0] # total number of observations
            self.p = p = x.shape[1] # number of covariates
            
        x, labels = any_to_tensor_and_labels(x, labels) # might need to do this for the other variables 
        
        ## now for secondary delta for the gamma frac
        if data_tensor == True:
            
            # Create tensor variable for the gamma_frac component of the likelihood
            
            self.event_change = event_change = theano.shared(np.array([np.append(np.repeat(1, s), np.repeat(0, k-s)).tolist()\
                                                                       for s in np.sum(event.get_value(), axis = 1)]), borrow = True)
        else:
            self.event_change = event_change = np.array([np.append(np.repeat(1, s), np.repeat(0, k-s)).tolist()\
                                                         for s in np.sum(event, axis = 1)])
            
        ## Keep track of total size of the dataset, for minibatching 
        ## new 10.10.2018
        # If minibatch, then we need the x component to be a generator and not just a tensor 
        # by this step in the computation, X is already in tensor form 
        
        if minibatch >= 2: # kinda hacky but whatever, we can fix this later 
            # If we're using mini-batch, then we have to tell the inner workings to fix the MAP estimate 
            minibatch = int(minibatch) #just in case some n00b puts in a double/float here 
            x_mini = pm.Minibatch(data = x.get_value(), batch_size = minibatch) # make minibatch instance of the design matrix
            time_mini = pm.Minibatch(data = time.get_value(), batch_size = minibatch) # minibatch instance of the time array
            event_mini = pm.Minibatch(data = event.get_value(), batch_size = minibatch) # minibatch instance of the event array
            event_change_mini = pm.Minibatch(data = event_change.get_value(), batch_size = minibatch) # minibatch instance of the transformed event array
            
            ## assign self. attributes to later parameterize the logp function 
            
            self.x = x_mini
            self.time = time_mini
            self.event = event_mini
            self.event_change = event_change_mini
            
        else:
            # if not minibatching, just pass the tensors as they are
            self.x = x
            self.time = time
            self.event = event
            self.event_change = event_change
            
        # now we have x, shape and labels
        
        # init a list to store all of the parameters that go into our likelihood
        coeffs_all = list()
        lams = list()
        rhos = list()
        
        for level in range(k): # for each dimension, instantiate a covariate effect for each predictor
            
            labels_this = [s + "_"+str(level) for s in labels]

            coeffs_this = list()
            for name in labels_this:
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
                coeffs_this.append(v)
            coeffs_this = tt.stack(coeffs_this, axis=0)
            coeffs_all.append(coeffs_this)
            
            ### Now for the baseline hazard portions 

            lam_name = 'lam_'+str(level)
            lam = self.Var(
                name = lam_name,
                dist = priors.get(lam_name,
                                  priors.get('lam', self.default_lambda_prior)
                                 )
            )# create labels for the lambdas
            lams.append(lam)
            # rhos
            rho_name = 'rho_'+str(level)
            rho = self.Var(
                name = rho_name,
                dist = priors.get(rho_name,
                                  priors.get('rho', self.default_rho_prior)
                                 )
            )
            rhos.append(rho)
            
            # finally, transformation parameters r
        # frailty parameter 
        theta = self.Var(
                name = 'theta',
                dist = priors.get('theta',
                                  priors.get('Theta', self.default_theta_prior)
                                 ))
        # make self attribute for the coefficients 
        self.coeffs_all = coeffs_all
        
        # changing 10.18 
        self.theta = theta
        self.lams = lams = tt.stack(lams, axis = 0)
        self.rhos = rhos = tt.stack(rhos, axis = 0)        


## Recommended use with VI

class Frailty_FixMAP(FrailtyIndependentComponent_Fix):
    """
    General class for inference of transformation frailty model. Inherits from FrailtIndependentComponent.
    
    Parameters
    ----------
    name : str - name, associated with the linear component
    x : tensor, covariate informatio (n x p dimension), if minibatched, then this is a generator
    time : tensor, time variable of (n x k dimension), if minibatched, then this is a generator
    event: tensor, event indicator variable of (n x k dimension), if minibatched, then this is a generator
    labels : list - replace variable names with these labels
    priors : dict - priors for coefficients
        use `Intercept` key for defining Intercept prior
            defaults to Flat.dist()
        use `Regressor` key for defining default prior for all regressors
            defaults to Normal.dist(mu=0, tau=1.0E-6)
    vars : dict - random variables instead of creating new ones
    """
    def __init__(self, time, event, x, rs, minibatch ='', family = 'gamma' ,labels=None,
                 priors=None, vars=None, name='', model=None):
        super(Frailty_FixMAP, self).__init__(time, event, x, minibatch, labels=labels,
            priors=priors, vars=vars, name=name, model=model
        )
        
        _families = dict(

            gamma = GammaFrailty
            
        )
        if isinstance(family, str):
            family = _families[family]()
        
        self.meow_rs = meow_rs = rs
        
        self.y_est = family.create_likelihood(name='', coeffs_all=self.coeffs_all, 
                                              rhos = self.rhos, 
                                              lams = self.lams, 
                                              rs = self.meow_rs,
                                              theta = self.theta,
                                              time=self.time, event = self.event, event_change = self.event_change, x = self.x,
                                              total_size = self.n,
                                              k = self.k,
                                              model=self)
        
        
        
parsurv = ParSurv
copula = Copula
frailty = Frailty
frailty_fixMAP = Frailty_FixMAP
