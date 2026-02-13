# Bayesian Inference Methods for Structural Dynamics
This repository contains core implementations of advanced Bayesian inference algorithms applied to structural damage identification. The project focuses on estimating stiffness reduction parameters in engineering structures through stochastic sampling.

## Key Implementations
* **Hamiltonian Monte Carlo (HMC):** A gradient-based sampling approach using a Leapfrog integrator for efficient exploration of high-dimensional posteriors.
* **Transitional MCMC (TMCMC):** A multi-level sampling strategy designed for complex, multi-modal probability landscapes.
* **Metropolis-Hastings (MH):** Baseline MCMC implementations featuring both Gaussian and Uniform proposal distributions.

## Project Context
These algorithms were developed to identify structural damage by comparing simulated nodal displacements with noisy synthetic measurements. 

*Note: The forward FEM solver and assembly modules are part of a private framework and are not included in this public repository.*
