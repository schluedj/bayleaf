### Likelihood storage for bayleaf
### Author: David Schlueter
### Vanderbilt University Department of Biostatistics
### July 10, 2017

import theano.tensor as tt
import numpy as np
import theano.tensor as tt

from pymc3.distributions import Continuous, draw_values, generate_samples, Bound, transforms


## Base class from pymc3
class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions"""
    def __init__(self, transform=transforms.log, *args, **kwargs):
        super(PositiveContinuous, self).__init__(
            transform=transform, *args, **kwargs)

################################################################################
###################### Univariate Parametric Models ############################
################################################################################
class Exponential_Censored(PositiveContinuous):
    """
    Exponential censored log-likelihood.
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------

    alpha : float
        For exponential model, set = 1 .
    """

    def __init__(self, alpha, indep, *args, **kwargs):
        super(Exponential_Censored, self).__init__(*args, **kwargs)
        self.indep = indep = tt.as_tensor_variable(indep)

    def logp(self, value, event):
        indep = self.indep
        indep = tt.exp(indep)
        return event * tt.log(indep) - indep * value

class Weibull_Censored(PositiveContinuous):
    """
    Weibull censored log-likelihood.
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    """

    def __init__(self, alpha, indep, *args, **kwargs):
        super(Weibull_Censored, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.indep = indep = tt.as_tensor_variable(indep)

    def logp(self, value, event):
        indep = self.indep
        alpha = self.alpha
        indep = tt.exp(indep)
        return event*(tt.log(alpha) + tt.log(indep) + (alpha-1)*tt.log(value))- (indep * value**alpha)

## CoxPH w/ weibull baseline hazard
class WeibullPH(PositiveContinuous):
    """
    Cox PH censored log-likelihood with weibull baseline hazard
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    """
    def __init__(self, alpha, lam, indep, *args, **kwargs):
        super(WeibullPH, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.indep = indep = tt.as_tensor_variable(indep)

    # Weibull survival likelihood, accounting for censoring
    def logp(self, value, event):
        indep = self.indep
        alpha = self.alpha
        lam = self.lam
        indep = tt.exp(indep)
        return event*(tt.log(alpha) + tt.log(lam) + tt.log(indep) + (alpha-1)*tt.log(value)) - (lam*indep * value**alpha)

class ExtremeValue_Censored(PositiveContinuous):
    
        """
        Extreme Value censored log-likelihood.
        .. math::
        ========  ====================================================
        ========  ====================================================
        Parameters
        ----------
        alpha : float
            Shape parameter (alpha > 0).
        """
        
    def __init__(self, alpha, indep, *args, **kwargs):
        super(ExtremeValue_Censored, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.indep = indep = tt.as_tensor_variable(indep)

    # Extreme Value survival likelihood, accounting for censoring
    def logp(self, value, event):
        indep = self.indep
        alpha = self.alpha
        return event*(tt.log(alpha)+(alpha*value)+indep) - tt.exp(indep+alpha*value)

#### TO ADD:  Gamma, Log-Normal

################################################################################
###################### Univariate Semi-Parametric Models ############################
################################################################################

#### To Add, Piecewise exponential


###############################################################################
###################### Multivariate Parametric Models ############################
################################################################################

#### To Add, Gamma frailty with Weibull Baseline hazard
