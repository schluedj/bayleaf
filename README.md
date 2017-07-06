# bayleaf
BAYesian Lifetime Event Analysis Framework

bayleaf is a Python package for Bayesian survival analysis using PyMC3.

Features
========
`bayleaf` will include:
-  Univariate Models: Parametric and Semi-Parametric Survival Models
-  Mulivariate Models: Parametric and Semi-Parametric Frailty Models 
-  Transformation Models
-  Flexible Survival Models: Gaussian Process and Spline Regression Survival Models 
-  Intuitive model specification syntax
-  Simulation functions from a general class of survival models 
-  By leveraging PyMC3, bayleaf models access modern Bayesian computation including:
   1. **Markov Chain Monte Carlo**, such as the [No U-Turn Sampler](http://www.jmlr.org/papers/v15/hoffman14a.html)
    2. **Variational inference**: [Automatic Differentiation Variational Inference](http://www.jmlr.org/papers/v18/16-107.html)
    for fast approximate posterior estimation as well as mini-batch ADVI
    for large data sets.
- Graphical outputs for survival models

Installation
------------

* from github (assumes bleeding edge pymc3 installed)
    ```bash
    # pip install git+git://github.com/pymc-devs/pymc3.git
    pip install git+https://github.com/schluedj/bayleaf.git
    ```

