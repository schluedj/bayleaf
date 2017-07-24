# bayleaf
BAYesian Lifetime Event Analysis Framework

### [Currently Under Construction]

bayleaf is a Python package for Bayesian survival analysis using [PyMC3](https://github.com/pymc-devs/pymc3).

Features ("imp": implemented and pushed, "comp": completed and about to be pushed, : "ip": in progress)
========
`bayleaf` will include:
-  Univariate Models: 
    1. Parametric: Weibull (imp), Exponential (imp), Extreme Value (imp)
    2. Semi-parametric: Piecewise hazard (comp) with Poisson Approximation
-  Multivariate Models: Parametric and Semi-Parametric Frailty Models
    1. Gamma Frailty Model with Weibull Baseline Hazard (comp)
-  Transformation Models:
    1. Transformed Weibull (comp)
    2. Transformed Piecewise Hazard (comp)
-  Flexible Survival Models: 
    1. Gaussian Process Survival Models (ip)
    2. Spline Regression Survival Models (comp)
-  Intuitive model specification syntax
-  Simulation functions from a general class of survival models
    1. Univariate Proportional Hazards with exponential censoring: Weibull (imp), Exponential (imp)
    2. Univariate Transformation Models (imp)
    3. Multivariate Transformation Frailty data (imp)
-  With a PyMC3 backend, bayleaf models leverage modern Bayesian computation including:
   1. **Markov Chain Monte Carlo**, such as the [No U-Turn Sampler](http://www.jmlr.org/papers/v15/hoffman14a.html)
    2. **Variational inference**: [Automatic Differentiation Variational Inference](http://www.jmlr.org/papers/v18/16-107.html)
    for fast approximate posterior estimation as well as mini-batch ADVI
    for large data sets.
- Graphical outputs for survival models

Installation
------------

* from github
    ```bash
    # pip install git+git://github.com/pymc-devs/pymc3.git
    pip install git+https://github.com/schluedj/bayleaf.git
    ```
