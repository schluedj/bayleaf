### Likelihood storage for bayleaf
### Author: David Schlueter
### Vanderbilt University Department of Biostatistics
### July 10, 2018

import theano.tensor as tt
import numpy as np

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

#class ExtremeValue_Censored(PositiveContinuous):

#        """
#        Extreme Value censored log-likelihood.
#        .. math::
#        ========  ====================================================
#        ========  ====================================================
#        Parameters
##        ----------
#        alpha : float
#            Shape parameter (alpha > 0).
#        """

#    def __init__(self, alpha, indep, *args, **kwargs):
#        super(ExtremeValue_Censored, self).__init__(*args, **kwargs)
#        self.alpha = alpha = tt.as_tensor_variable(alpha)
#        self.indep = indep = tt.as_tensor_variable(indep)

    # Extreme Value survival likelihood, accounting for censoring
#    def logp(self, value, event):
#        indep = self.indep
#        alpha = self.alpha
#        return event*(tt.log(alpha)+(alpha*value)+indep) - tt.exp(indep+alpha*value)

#### TO ADD:  Gamma, Log-Normal

################################################################################
###################### Univariate Semi-Parametric Models ############################
################################################################################

#### To Add, Piecewise exponential


###############################################################################
###################### Multivariate Parametric Models ############################
################################################################################

#### To Add, Gamma frailty with Weibull Baseline hazard

###############################################################################
###################### Multivariate Parametric Models ##########################
################################################################################
############################# Copula Likelihoods ###############################


class Clayton_Censored(PositiveContinuous):
    """
    Bivariate Clayton Censored Model
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha: 
        association parameter between dimensions
    indep_1, indep_2: 
        independent components, i.e. covariate effects
    lam_1, lam2, rho_1, rho_2: 
        >0 weibull baseline hazard parameters 
    r_1, r_2: 
        transformation parameters 

    
    """
    def __init__(self, alpha, indep_1, indep_2, rho_1, lam_1, rho_2, lam_2,
                 *args, **kwargs):
        super(Clayton_Censored, self).__init__(*args, **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        self.indep_1 = indep_1 = tt.as_tensor_variable(indep_1)
        self.lam_1 = lam_1 = tt.as_tensor_variable(lam_1)
        self.rho_1 = rho_1 = tt.as_tensor_variable(rho_1)

        self.indep_2 = indep_2 = tt.as_tensor_variable(indep_2)
        self.lam_2 = lam_2 = tt.as_tensor_variable(lam_2)
        self.rho_2 = rho_2 = tt.as_tensor_variable(rho_2)

    def logp(self, time_1, time_2, delta_1, delta_2):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event indicator in the second dimension.

        """
        ## define local instances of the globally initiated variables
        alpha = self.alpha

        indep_1 = self.indep_1
        lam_1 = self.lam_1
        rho_1 = self.rho_1

        indep_2 = self.indep_2
        lam_2 = self.lam_2
        rho_2 = self.rho_2

        ### Now define survival quantities
        ### Baseline quantities
        # H(t) = lam*t^{rho}
        base_cum_hazard_1 = lam_1*time_1**(rho_1)
        base_cum_hazard_2 = lam_2*time_2**(rho_2)

        # h(t) = lam*rho*t^{rho-1}
        base_hazard_1 = lam_1*rho_1*time_1**(rho_1-1)
        base_hazard_2 = lam_2*rho_2*time_2**(rho_2-1)

        # h(t|X) = h(t)*exp(X'β)
        conditional_hazard_1 = base_hazard_1 * tt.exp(indep_1)
        conditional_hazard_2 = base_hazard_2 * tt.exp(indep_2)
        # H(t|X) = H(t)*exp(X'β)
        conditional_cum_hazard_1 = base_cum_hazard_1 * tt.exp(indep_1)
        conditional_cum_hazard_2 = base_cum_hazard_2 * tt.exp(indep_2)

        # S(t|X) = exp(-H(t|X))
        surv_1 = tt.exp(-conditional_cum_hazard_1)
        surv_2 = tt.exp(-conditional_cum_hazard_2)

        ## f(t|X) = S(t|X)*h(t|X)
        density_1 = conditional_hazard_1 * surv_1
        density_2 = conditional_hazard_2 * surv_2

        ### Copula derivatives:
        ### Copula derivatives:
        log_clayton_copula = (-alpha)**(-1)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d_clayton_copula_s1 = -(alpha+1)*tt.log(surv_1)-((alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d_clayton_copula_s2 = -(alpha+1)*tt.log(surv_2)-((alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d2_clayton_copula_s1_s2 = tt.log(alpha+1)+(-(alpha+1))*tt.log(surv_1*surv_2)-((2*alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        ### different parts of log likelihood
        first = delta_1*delta_2*(log_d2_clayton_copula_s1_s2+tt.log(density_1)+tt.log(density_2))
        second = delta_1*(1-delta_2)*(log_d_clayton_copula_s1+tt.log(density_1))
        third = delta_2*(1-delta_1)*(log_d_clayton_copula_s2+tt.log(density_2))
        fourth = (1-delta_1)*(1-delta_2)*log_clayton_copula

        return first + second + third + fourth

class Clayton_Censored_Trans(PositiveContinuous):
    """
    Bivariate Clayton Censored Copula Model
    ## This function comes from the author's dissertation. Paper forthcoming.
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha: 
        association parameter between dimensions
    indep_1, indep_2: 
        independent components, i.e. covariate effects
    lam_1, lam2, rho_1, rho_2: >0 
        weibull baseline hazard parameters 
    r_1, r_2: 
        transformation parameters 

    """
    def __init__(self, alpha, indep_1, indep_2, rho_1, lam_1, rho_2, lam_2, r_1, r_2,
                 *args, **kwargs):
        super(Clayton_Censored_Trans, self).__init__(*args, **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        ## Parameters for first dimension
        self.indep_1 = indep_1 = tt.as_tensor_variable(indep_1)
        self.lam_1 = lam_1 = tt.as_tensor_variable(lam_1)
        self.rho_1 = rho_1 = tt.as_tensor_variable(rho_1)
        self.r_1  = r_1 = tt.as_tensor_variable(r_1)

        ## Parameters for second dimension
        self.indep_2 = indep_2 = tt.as_tensor_variable(indep_2)
        self.lam_2 = lam_2 = tt.as_tensor_variable(lam_2)
        self.rho_2 = rho_2 = tt.as_tensor_variable(rho_2)
        self.r_2  = r_2 = tt.as_tensor_variable(r_2)

    def logp(self, time_1, time_2, delta_1, delta_2):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event indicator in the second dimension.

        """
        ## define local instances of the globally initiated variables
        alpha = self.alpha

        indep_1 = self.indep_1
        lam_1 = self.lam_1
        rho_1 = self.rho_1
        r_1 = self.r_1

        indep_2 = self.indep_2
        lam_2 = self.lam_2
        rho_2 = self.rho_2
        r_2 = self.r_2

        ### Now define survival quantities
        ### Baseline quantities
        # H(t) = lam*t^{rho}
        base_cum_hazard_1 = lam_1*time_1**(rho_1)
        base_cum_hazard_2 = lam_2*time_2**(rho_2)

        # h(t) = lam*rho*t^{rho-1}
        base_hazard_1 = lam_1*rho_1*time_1**(rho_1-1)
        base_hazard_2 = lam_2*rho_2*time_2**(rho_2-1)

        # h(t|X) = h(t)*exp(X'β)
        #conditional_hazard_1 = base_hazard_1 * tt.exp(indep_1)
        #conditional_hazard_2 = base_hazard_2 * tt.exp(indep_2)
        # H(t|X) = log(1+r*H(t)*exp(X'β))/r
        conditional_cum_hazard_1 = tt.log(1 + r_1 * base_cum_hazard_1 * tt.exp(indep_1))/r_1
        conditional_cum_hazard_2 = tt.log(1 + r_2 * base_cum_hazard_2 * tt.exp(indep_2))/r_2

        # S(t|X) = exp(-H(t|X))
        surv_1 = tt.exp(-conditional_cum_hazard_1)
        surv_2 = tt.exp(-conditional_cum_hazard_2)

        ## f(t|X) = S(t|X)*h(t|X)
        density_1 = base_hazard_1*tt.exp(indep_1)*(1+r_1*base_cum_hazard_1*tt.exp(indep_1))**-(1+r_1**(-1))
        density_2 = base_hazard_2*tt.exp(indep_2)*(1+r_2*base_cum_hazard_2*tt.exp(indep_2))**-(1+r_2**(-1))

        ### Copula derivatives:
        log_clayton_copula = (-alpha)**(-1)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d_clayton_copula_s1 = -(alpha+1)*tt.log(surv_1)-((alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d_clayton_copula_s2 = -(alpha+1)*tt.log(surv_2)-((alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        log_d2_clayton_copula_s1_s2 = tt.log(alpha+1)+(-(alpha+1))*tt.log(surv_1*surv_2)-((2*alpha+1)/alpha)*tt.log(surv_1**(-alpha)+surv_2**(-alpha)-1)
        ### different parts of log likelihood
        first = delta_1*delta_2*(log_d2_clayton_copula_s1_s2+tt.log(density_1)+tt.log(density_2))
        second = delta_1*(1-delta_2)*(log_d_clayton_copula_s1+tt.log(density_1))
        third = delta_2*(1-delta_1)*(log_d_clayton_copula_s2+tt.log(density_2))
        fourth = (1-delta_1)*(1-delta_2)*log_clayton_copula

        return first + second + third + fourth

class Joe_Censored_Trans(PositiveContinuous):
    """
    Bivariate Joe Censored Model
    ## This function comes from the author's dissertation. Paper forthcoming.
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha: 
        association parameter between dimensions
    indep_1, indep_2: 
        independent components, i.e. covariate effects
    lam_1, lam2, rho_1, rho_2: 
        >0 weibull baseline hazard parameters 
    r_1, r_2: 
        transformation parameters 
    
    """
    def __init__(self, alpha, indep_1, indep_2, rho_1, lam_1, rho_2, lam_2, r_1, r_2,
                 *args, **kwargs):
        super(Joe_Censored_Trans, self).__init__(*args, **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        ## Parameters for first dimension
        self.indep_1 = indep_1 = tt.as_tensor_variable(indep_1)
        self.lam_1 = lam_1 = tt.as_tensor_variable(lam_1)
        self.rho_1 = rho_1 = tt.as_tensor_variable(rho_1)
        self.r_1  = r_1 = tt.as_tensor_variable(r_1)

        ## Parameters for second dimension
        self.indep_2 = indep_2 = tt.as_tensor_variable(indep_2)
        self.lam_2 = lam_2 = tt.as_tensor_variable(lam_2)
        self.rho_2 = rho_2 = tt.as_tensor_variable(rho_2)
        self.r_2  = r_2 = tt.as_tensor_variable(r_2)

    def logp(self, time_1, time_2, delta_1, delta_2):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event indicator in the second dimension.

        """
        ## define local instances of the globally initiated variables
        alpha = self.alpha
        #alpha = self.alpha = alpha + 1
        indep_1 = self.indep_1
        lam_1 = self.lam_1
        rho_1 = self.rho_1
        r_1 = self.r_1

        indep_2 = self.indep_2
        lam_2 = self.lam_2
        rho_2 = self.rho_2
        r_2 = self.r_2

        ### Now define survival quantities
        ### Baseline quantities
        # H(t) = lam*t^{rho}
        base_cum_hazard_1 = lam_1*time_1**(rho_1)
        base_cum_hazard_2 = lam_2*time_2**(rho_2)

        # h(t) = lam*rho*t^{rho-1}
        base_hazard_1 = lam_1*rho_1*time_1**(rho_1-1)
        base_hazard_2 = lam_2*rho_2*time_2**(rho_2-1)

        # h(t|X) = h(t)*exp(X'β)
        #conditional_hazard_1 = base_hazard_1 * tt.exp(indep_1)
        #conditional_hazard_2 = base_hazard_2 * tt.exp(indep_2)
        # H(t|X) = log(1+r*H(t)*exp(X'β))/r
        conditional_cum_hazard_1 = tt.log(1 + r_1 * base_cum_hazard_1 * tt.exp(indep_1))/r_1
        conditional_cum_hazard_2 = tt.log(1 + r_2 * base_cum_hazard_2 * tt.exp(indep_2))/r_2

        # S(t|X) = exp(-H(t|X))
        surv_1 = tt.exp(-conditional_cum_hazard_1)
        surv_2 = tt.exp(-conditional_cum_hazard_2)

        ## f(t|X) = S(t|X)*h(t|X)
        density_1 = base_hazard_1*tt.exp(indep_1)*(1+r_1*base_cum_hazard_1*tt.exp(indep_1))**-(1+r_1**(-1))
        density_2 = base_hazard_2*tt.exp(indep_2)*(1+r_2*base_cum_hazard_2*tt.exp(indep_2))**-(1+r_2**(-1))

        ### Copula derivatives:
        log_joe_copula = tt.log(1-((1-surv_1)**alpha+(1-surv_2)**alpha-((1-surv_1)**alpha)*((1-surv_2)**alpha))**(1/alpha))
        #log_joe_copula =
        log_d_joe_copula_s1 =tt.log((-surv_1+1)**(alpha-1)*(-(-surv_2+1)**alpha+1)*(-(-surv_1+1)**alpha*(-surv_2+1)**alpha + (-surv_1+1)**alpha + (-surv_2+1)**alpha)**((-alpha +1)/alpha))

        log_d_joe_copula_s2 =tt.log((-surv_2 + 1)**(alpha - 1)*(-(-surv_1 + 1)**alpha + 1)*(-(-surv_1 + 1)**alpha*(-surv_2 + 1)**alpha + (-surv_1 + 1)**alpha + (-surv_2 + 1)**alpha)**((-alpha + 1)/alpha))

        log_d2_joe_copula_s1_s2 =tt.log((-surv_1 + 1)**(alpha - 1)*(-surv_2 + 1)**(alpha - 1)*(-(-surv_1 + 1)**alpha*(-surv_2 + 1)**alpha + (-surv_1 + 1)**alpha + (-surv_2 + 1)**alpha)**(-2 + 1/alpha)*\
                                    (alpha*((-surv_1 + 1)**alpha - 1)*((-surv_2 + 1)**alpha - 1) + alpha*(-(-surv_1 + 1)**alpha*(-surv_2 + 1)**alpha + (-surv_1 + 1)**alpha + (-surv_2 + 1)**alpha) - ((-surv_1 + 1)**alpha - 1)\
                                     *((-surv_2 + 1)**alpha - 1)))
        first = delta_1*delta_2*(log_d2_joe_copula_s1_s2+tt.log(density_1)+tt.log(density_2))
        second = delta_1*(1-delta_2)*(log_d_joe_copula_s1+tt.log(density_1))
        third = delta_2*(1-delta_1)*(log_d_joe_copula_s2+tt.log(density_2))
        fourth = (1-delta_1)*(1-delta_2)*log_joe_copula

        return first + second + third + fourth

class Frank_Censored_Trans(PositiveContinuous):
    """
    Bivariate Frank Censored Copula Model
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha: association parameter between dimensions
    indep_1, indep_2: independent components, i.e. covariate effects
    lam_1, lam2, rho_1, rho_2: >0 weibull baseline hazard parameters 
    r_1, r_2: transformation parameters 

    """
    def __init__(self, alpha, indep_1, indep_2, rho_1, lam_1, rho_2, lam_2, r_1, r_2,
                 *args, **kwargs):
        super(Frank_Censored_Trans, self).__init__(*args, **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        ## Parameters for first dimension
        self.indep_1 = indep_1 = tt.as_tensor_variable(indep_1)
        self.lam_1 = lam_1 = tt.as_tensor_variable(lam_1)
        self.rho_1 = rho_1 = tt.as_tensor_variable(rho_1)
        self.r_1  = r_1 = tt.as_tensor_variable(r_1)

        ## Parameters for second dimension
        self.indep_2 = indep_2 = tt.as_tensor_variable(indep_2)
        self.lam_2 = lam_2 = tt.as_tensor_variable(lam_2)
        self.rho_2 = rho_2 = tt.as_tensor_variable(rho_2)
        self.r_2  = r_2 = tt.as_tensor_variable(r_2)

    def logp(self, time_1, time_2, delta_1, delta_2):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event indicator in the second dimension.

        """
        ## define local instances of the globally initiated variables
        alpha = self.alpha

        indep_1 = self.indep_1
        lam_1 = self.lam_1
        rho_1 = self.rho_1
        r_1 = self.r_1

        indep_2 = self.indep_2
        lam_2 = self.lam_2
        rho_2 = self.rho_2
        r_2 = self.r_2

        ### Now define survival quantities
        ### Baseline quantities
        # H(t) = lam*t^{rho}
        base_cum_hazard_1 = lam_1*time_1**(rho_1)
        base_cum_hazard_2 = lam_2*time_2**(rho_2)

        # h(t) = lam*rho*t^{rho-1}
        base_hazard_1 = lam_1*rho_1*time_1**(rho_1-1)
        base_hazard_2 = lam_2*rho_2*time_2**(rho_2-1)

        # h(t|X) = h(t)*exp(X'β)
        #conditional_hazard_1 = base_hazard_1 * tt.exp(indep_1)
        #conditional_hazard_2 = base_hazard_2 * tt.exp(indep_2)
        # H(t|X) = log(1+r*H(t)*exp(X'β))/r
        conditional_cum_hazard_1 = tt.log(1 + r_1 * base_cum_hazard_1 * tt.exp(indep_1))/r_1
        conditional_cum_hazard_2 = tt.log(1 + r_2 * base_cum_hazard_2 * tt.exp(indep_2))/r_2

        # S(t|X) = exp(-H(t|X))
        surv_1 = tt.exp(-conditional_cum_hazard_1)
        surv_2 = tt.exp(-conditional_cum_hazard_2)

        ## f(t|X) = S(t|X)*h(t|X)
        density_1 = base_hazard_1*tt.exp(indep_1)*(1+r_1*base_cum_hazard_1*tt.exp(indep_1))**-(1+r_1**(-1))
        density_2 = base_hazard_2*tt.exp(indep_2)*(1+r_2*base_cum_hazard_2*tt.exp(indep_2))**-(1+r_2**(-1))

        ### Copula derivatives:
        log_frank_copula = tt.log(-(1/alpha)*tt.log(1+((tt.exp(-alpha*surv_1)-1)*(tt.exp(-alpha*surv_2)-1))/(tt.exp(-alpha)-1)))
        log_d_frank_copula_s1 = tt.log((tt.exp(surv_2*alpha) - 1)*tt.exp(alpha)/((tt.exp(alpha) - 1)*tt.exp(alpha*(surv_1 + surv_2)) - (tt.exp(surv_1*alpha) - 1)*(tt.exp(surv_2*alpha) - 1)*tt.exp(alpha)))
        log_d_frank_copula_s2 = tt.log((tt.exp(surv_1*alpha) - 1)*tt.exp(alpha)/((tt.exp(alpha) - 1)*tt.exp(alpha*(surv_1 + surv_2)) - (tt.exp(surv_1*alpha) - 1)*(tt.exp(surv_2*alpha) - 1)*tt.exp(alpha)))
        log_d2_frank_copula_s1_s2 = tt.log(alpha*(1 + (1 - tt.exp(-surv_1*alpha))*(1 - tt.exp(-surv_2*alpha))/((1 - (1 - tt.exp(-surv_1*alpha))*(1 - tt.exp(-surv_2*alpha))/(1 - tt.exp(-alpha)))*(1 - tt.exp(-alpha))))*tt.exp(-surv_1*alpha)*tt.exp(-surv_2*alpha)/((1 - (1 - tt.exp(-surv_1*alpha))*(1 - tt.exp(-surv_2*alpha))/(1 - tt.exp(-alpha)))*(1 - tt.exp(-alpha))))

        ### different parts of log likelihood

        first = delta_1*delta_2*(log_d2_frank_copula_s1_s2+tt.log(density_1)+tt.log(density_2))
        second = delta_1*(1-delta_2)*(log_d_frank_copula_s1+tt.log(density_1))
        third = delta_2*(1-delta_1)*(log_d_frank_copula_s2+tt.log(density_2))
        fourth = (1-delta_1)*(1-delta_2)*log_frank_copula

        return first + second + third + fourth


class Gumbel_Censored_Trans(PositiveContinuous):
    """
    Bivariate Gumbel Censored Model
    ## This function comes from the Author's dissertation. Paper forthcoming.
    .. math::
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    alpha: 
        association parameter between dimensions
    indep_1, indep_2: 
        independent components, i.e. covariate effects
    lam_1, lam2, rho_1, rho_2: 
        >0 weibull baseline hazard parameters 
    r_1, r_2: 
        transformation parameters 

    """
    def __init__(self, alpha, indep_1, indep_2, rho_1, lam_1, rho_2, lam_2, r_1, r_2,
                 *args, **kwargs):
        super(Gumbel_Censored_Trans, self).__init__(*args, **kwargs)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        ## Parameters for first dimension
        self.indep_1 = indep_1 = tt.as_tensor_variable(indep_1)
        self.lam_1 = lam_1 = tt.as_tensor_variable(lam_1)
        self.rho_1 = rho_1 = tt.as_tensor_variable(rho_1)
        self.r_1  = r_1 = tt.as_tensor_variable(r_1)

        ## Parameters for second dimension
        self.indep_2 = indep_2 = tt.as_tensor_variable(indep_2)
        self.lam_2 = lam_2 = tt.as_tensor_variable(lam_2)
        self.rho_2 = rho_2 = tt.as_tensor_variable(rho_2)
        self.r_2  = r_2 = tt.as_tensor_variable(r_2)

    def logp(self, time_1, time_2, delta_1, delta_2):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event indicator in the second dimension.

        """
        ## define local instances of the globally initiated variables
        alpha = self.alpha
        #alpha = self.alpha = alpha+1
        indep_1 = self.indep_1
        lam_1 = self.lam_1
        rho_1 = self.rho_1
        r_1 = self.r_1

        indep_2 = self.indep_2
        lam_2 = self.lam_2
        rho_2 = self.rho_2
        r_2 = self.r_2

        ### Now define survival quantities
        ### Baseline quantities
        # H(t) = lam*t^{rho}
        base_cum_hazard_1 = lam_1*time_1**(rho_1)
        base_cum_hazard_2 = lam_2*time_2**(rho_2)

        # h(t) = lam*rho*t^{rho-1}
        base_hazard_1 = lam_1*rho_1*time_1**(rho_1-1)
        base_hazard_2 = lam_2*rho_2*time_2**(rho_2-1)

        # h(t|X) = h(t)*exp(X'β)
        #conditional_hazard_1 = base_hazard_1 * tt.exp(indep_1)
        #conditional_hazard_2 = base_hazard_2 * tt.exp(indep_2)
        # H(t|X) = log(1+r*H(t)*exp(X'β))/r
        conditional_cum_hazard_1 = tt.log(1 + r_1 * base_cum_hazard_1 * tt.exp(indep_1))/r_1
        conditional_cum_hazard_2 = tt.log(1 + r_2 * base_cum_hazard_2 * tt.exp(indep_2))/r_2

        # S(t|X) = exp(-H(t|X))
        surv_1 = tt.exp(-conditional_cum_hazard_1)
        surv_2 = tt.exp(-conditional_cum_hazard_2)

        ## f(t|X) = S(t|X)*h(t|X)
        density_1 = base_hazard_1*tt.exp(indep_1)*(1+r_1*base_cum_hazard_1*tt.exp(indep_1))**-(1+r_1**(-1))
        density_2 = base_hazard_2*tt.exp(indep_2)*(1+r_2*base_cum_hazard_2*tt.exp(indep_2))**-(1+r_2**(-1))

        ### Copula derivatives:
        log_gumbel_copula = -((-tt.log(surv_1))**(alpha)+(-tt.log(surv_2))**(alpha))**(1/alpha)
        log_d_gumbel_copula_s1 = (log_gumbel_copula -((alpha-1)/alpha)*tt.log((-tt.log(surv_1))**(alpha)+(-tt.log(surv_2))**(alpha))+(alpha-1)*np.log(-tt.log(surv_1))-tt.log(surv_1))
        log_d_gumbel_copula_s2 = (log_gumbel_copula -((alpha-1)/alpha)*tt.log((-tt.log(surv_1))**(alpha)+(-tt.log(surv_2))**(alpha))+(alpha-1)*np.log(-tt.log(surv_2))-tt.log(surv_2))
        log_d2_gumbel_copula_s1_s2 = log_gumbel_copula-(tt.log(surv_1)+tt.log(surv_2))+(2/alpha-2)*tt.log((-tt.log(surv_1))**(alpha)+(-tt.log(surv_2))**(alpha))+(alpha-1)*tt.log(np.log(surv_1)*tt.log(surv_2))+tt.log(1+(alpha-1)*((-tt.log(surv_1))**(alpha)+(-tt.log(surv_2))**(alpha))**(-1/alpha))

        first = delta_1*delta_2*(log_d2_gumbel_copula_s1_s2+tt.log(density_1)+tt.log(density_2))
        second = delta_1*(1-delta_2)*(log_d_gumbel_copula_s1+tt.log(density_1))
        third = delta_2*(1-delta_1)*(log_d_gumbel_copula_s2+tt.log(density_2))
        fourth = (1-delta_1)*(1-delta_2)*log_gumbel_copula
        return first + second + third + fourth

### Frailty likelihood 

class Gamma_Frailty(PositiveContinuous):
    """
    Likelihood for transformation Gamma frailty model 
    ========  ====================================================
    ========  ====================================================
    Parameters
    ----------
    theta: 
        gamma frailty parameter
    coeffs_all: 
        coefficients for covariate effects 
    lams, rhos: 
        >0 weibull baseline hazard parameters 
    rs: 
        transformation parameters
    k: 
        dimension of outcome
    """
    def __init__(self, theta, coeffs_all, rhos, lams, rs, k,
                 *args, **kwargs):
        super(Gamma_Frailty, self).__init__(*args, **kwargs)
        
        self.theta = theta = tt.as_tensor_variable(theta)
        self.coeffs_all = coeffs_all = tt.as_tensor_variable(coeffs_all)
        #self.linear = linear = tt.as_tensor_variable(linear)
        self.lams = lams = tt.as_tensor_variable(lams)
        self.rhos = rhos = tt.as_tensor_variable(rhos)
        self.rs = rs = tt.as_tensor_variable(rs)
        
        
        # Might be a little hacky but find dimension of populated tensor
        self.k = k
       
    def logp(self, time, delta_1, delta_2, x):
        """
        time_1: array
            time in the first dimension.
        time_2: array
            time in the second dimension.
        delta_1: array
            event indicator in the first dimension.
        delta_2: array
            event re-arrange for the vector needed in the gamma_frac
        x: array
            covariate information (independent component)

        """
        ## define local instances of the globally initiated variables
        theta = self.theta
        coeffs_all = self.coeffs_all
       
        lams = self.lams
        rhos = self.rhos
        rs = self.rs
        linear = tt.dot(coeffs_all, x.T).T
        gamma_frac = tt.dot(delta_2, tt.log(theta**(-1) + tt.arange(self.k)))
        #linear = tt.dot(β,X.T).T# this is the correct formulation
        weib_base_haz = lams*rhos*time**(rhos-1) #weib haz
        weib_base_cumhaz = lams*time**(rhos)  # cumulative ha
        phi_1 = tt.log(weib_base_haz*np.exp(linear))
        phi_2 = tt.log((1+rs*weib_base_cumhaz*np.exp(linear)))
        failed_component = tt.sum(delta_1*phi_1, axis = 1)-tt.sum(delta_1*phi_2, axis = 1)
        psi = tt.log(tt.sum(tt.log(1+rs*weib_base_cumhaz*tt.exp(linear))/rs,axis=1)+theta**(-1))
                    # second component for all the censored observations
        one_k = tt.ones(self.k)
        second = (theta**(-1)+tt.dot(delta_1, one_k))*psi
            # define log likelihood
        return gamma_frac + failed_component + theta**(-1)*tt.log(theta**(-1)) - second
