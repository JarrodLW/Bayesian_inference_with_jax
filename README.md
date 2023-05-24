
# A Bayesian optimisation toolkit built on jax

**Summary**: basic single-objective Bayesian Optimisation,
for scalar objectives defined over a domain of arbitrary dimension. Implementation using JAX allows for 
auto-differentiation of acquisition functions etc, enabling the use gradient-based approaches to optimisation, 
in particular the use of algorithms from the package Optax. Implementation allows for arbitrary function to
be passed as prior. Various kernels, acquisition functions and acquisition
algorithms implemented (see below). The top-level API (see *Experiment.py* script) allows for
easy incorporation of expert knowledge (e.g. concerning kernel types, defining hyperparameters, priors etc.), 
on the one hand, and the automatic fitting of unspecified hyperparameters, on the other. See the Jupyter file for 
examples. 

**Random Process Models**: only Gaussian Processes, so far. *See Regressors.py*.

**Kernels**: RBF, Periodic, Matern (orders 1/2, 3/2, 5/2) and any algebraic combination thereof 
---i.e. arbitrary products and linear combinations. *See Kernels.py*.
 
**Acquisition functions**: PI, EI, UCB and arbitrary linear combinations thereof. *See AcquisitionFuncs.py*.

**Algorithms for maximisation of acquisition function**: Random search, any optax optimiser. *See Algorithms.py*.

**Fitting of kernel hyperparameters**: maximum likelihood estimation (MLE) routine for fitting to data. API allows for 
partial prescription of kernel hyperparameters, the remaining hyperparameters being inferred by MLE (see 
*maximum_likelihood_estimation* wrapper method in the *Experiment* class). Note that this has only been implemented for 
the three basic kernel types, currently. *See Algorithms.py*.

