# bayleaf
BAYesian Lifetime Event Analysis Framework
###[Currently Under Construction]

bayleaf is a Python package for Bayesian survival analysis using [PyMC3](https://github.com/pymc-devs/pymc3).

Features
========
`bayleaf` will include:
-  Univariate Models: Parametric and Semi-Parametric Survival Models
-  Multivariate Models: Parametric and Semi-Parametric Frailty Models
-  Transformation Models
-  Flexible Survival Models: Gaussian Process and Spline Regression Survival Models
-  Intuitive model specification syntax
-  Simulation functions from a general class of survival models
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
