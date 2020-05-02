---
layout: post
author: Colin Swaney
title: Event-Driven Bayesian Models
date: 2020-04-25
categories: [research]
category: research
tags: [machine learning, high-frequency trading]
excerpt: "<p></p>"
---

## Network Models

### Spike-and-Slab
So-called "spike-and-slab" models separate concerns between the existence of connections in a network and the strength of those connections.  

### Erdos-Renyi Network
In the Erdos-Renyi model, each connection in the model is sampled independently from a $$Bernoulli(\rho)$$ distribution.

### Stochastic Block Network
In the Stochastic Block model, each node is assigned a class, and each connection in the model is sampled independently *conditional* on the node classes:

$$ A_{i, j} \sim Bernoulli(\rho_{z_i}, \rho_{z_j}), $$

where $$z_k$$ is the latent class of the $$k$$-th node.

### Latent Distance Network
In the Latent Distance Network, the distance between networks determines the probability of a connection between nodes.

## Point Processes
Point processes model sequences of events:

$$ \{s_m\}_{m=1}^M $$

- Typically, point processes model events in time, $$s_m \in \left[0, T\right]$$, but they could also model events in space, $$s_m \in \mathbb{R}^D$$.
- A *Poisson process* is a point process in which the probability of events is determined by an intensity function $$\lambda(t)$$ such that the number of events occuring in a period $$\left[t, t + \Delta t \right]$$ has a Poisson distribution:

$$ M \sim \text{Poisson}(\mu), $$

$$\mu = \int_{t}^{t + \Delta t} \lambda(t) dt $$

- The number of events in non-overlapping periods $$\mathcal{V}$$ and $$\mathcal{W}$$ are independent.
- A general process for simulating a Poisson process consists of sampling a number of events according to the Poisson distribution above, then sampling the time of the events according to the distribution of $$\lambda(t)$$ throuhout $$\left[0, T\right]$$:

$$ s_m \sim p(s) = \frac{\lambda(t)}{\int_0^T \lambda(t) dt}$$

- The likelihood of a sequence of events generated according to a Poisson process is calculated accordingly. First, calculate the probability of observing $$M$$ events:

$$ p(M) = \frac{\lambda^{M}e^{-\int_0^T \lambda(t) dt}}{M!} $$

- Next, calculate the likelihood of the *collection* $$\{s_m\}_{m=1}^M$$:

$$ \left( \prod_{m=1}^M \frac{\lambda(s_m)}{\int_0^T \lambda(t) dt} \right) M! $$

- Notice that these events are not "ordered", thus we multiply by $$M!$$ to account for all the possible ways of observing the times in $$\{s_m\}_{m=1}^M$$.

- After simplifying, the overall likelihood is given by

$$ L(\{s_m\} \ | \ \lambda) = \exp \left(-\int_0^T \lambda(t) dt \right) \prod_{m=1}^M \lambda(s_m) $$

### Homogeneous Processes
- In the simplest model, the intensity is constant: $$\lambda(t) \equiv \lambda$$. In this case the location of events in the simulation process has a uniform distribution, and the likelihood function simplifies to

$$ L(\{s_m\} \ | \ \lambda) = \exp \left(- \lambda T \right) \lambda^M $$

![homogeneous-poisson-process-likelihood](/assets/img/homogeneous-poisson-process-likelihood.svg)

- The chart above displays the likelihood as a function of $$\lambda$$ for sequence that consists of ten events occuring over the period $$[0, 10]$$.
- Notice that for a homogeneous Poisson process, the timing of events is unimportant because they are uniformly distributed throughout time.

### Hawkes Processes
Homogeneous Poisson processes wont get us too far. In interesting applications, the probability of events changes over time in response to environmental factors, perhaps dramatically so. We can model such environments through time-varying intensity functions augmented by environmental variables:

$$ \lambda(t) = \lambda(t, x_t) $$

- For some processes, the intensity is better characterized as "self-exciting": when one event occurs, it *causes* additional events. Hawkes processes are a class of point processes with this property.

#### Model

$$ \lambda(t) = \lambda^{(0)} + \sum_{s_m < t} w \cdot \theta e^{-\theta (t - s_m)} $$

- With this formulation, each event directly causes $$W$$ events in expectation because $$ \theta e^{-\theta (t - s_m)} $$ is proper probability distribution. If no events occur, we expect $$\lambda^{(0)}$$ events per unit of time.
- The effect of each event is infinitely long, but its relevance decays exponentially.

#### Simulation
- Hawkes models also permit a straight-forward generative model, which relies on an interesting property of Poisson processes.

#### The Poisson Superposition Principle
Simply stated, the Poisson superposition principle states that there is no observational difference between a collection of independent Poisson processes and a single Poisson process whose intensity equals the sum of the intensities in the collection.

$$ \lambda_{tot}(t) = \sum_{k=1}^K \lambda_k(t) $$

$$ L(\{s_m\}_{m=1}^M \ | \ \lambda_{tot}) = L(\{ \{s_m\}_{m=1}^{M_k} \}_{k=1}^K \ | \ \{ \lambda_k \}_{k=1}^K) $$

- The likelihood of a univariate Hawkes is calculated exactly as stated above.

### Multivariate Hawkes Processes
In a multivariate, $$N$$ processes generate events, and each event amplifies the intensity of every process. The processes are *dependent*, which complicates the calculation of likelihood, but allows us to model interactions between processes.

#### Model

$$ \lambda_n(t) = \lambda_n^{(0)} + \sum_{s_m < t} W_{c_m, n} \cdot \theta_{c_m, n} \cdot e^{-\theta_{c_m, n} (t - s_m)} $$

#### Simulation
- We can simulate a multivariate Hawkes process using a similar approach based on the Poisson Superposition Principle. The only difference in the multivariate case is that each event spawns an (independent) Poisson process on each of the other $$N$$ channels.

#### Likelihood
- We can use the Poisson Superposition Principle to calculate likelihood. The principle requires *independent* processes. If we observe the "parent" of each event, $$\omega_m$$, then we can calculate the likelihood by computing $$\lambda_{tot}(s_m)$$ at each $$s_m$$ and applying the Poisson likelihood formula (noting that the integral of the exponential distribution is easy to compute).
- In practice, $$\omega_m$$ are latent variables. But the likelihood of the events can be obtained by marginalizing over the parent variables. That is, by summing over all possible parents at each event (including the background process on the given node).
- Applying this to each of the $$N$$ processes, we get

$$ L(\{s_m, c_m\} \ | \ \theta, W) = \dots $$


## Combining Network Models and Point Processes
To combine the spike-and-slab network model with the multivariate Hawkes process we simply modify the impulse-response to include the "spike" parameters, $$A_{i, j}$$.

$$ \lambda_n(t) = \lambda_n^{(0)} + \sum_{s_m < t} A_{c_m, n} \cdot W_{c_m, n} \cdot \theta_{c_m, n} \cdot e^{-\theta_{c_m, n} (t - s_m)}, $$

where $$A_{c_m, n} \in \{0, 1\}$$.

- The network model and the temporal model are *almost* disjoint: they connect through the influence of $$A$$ on the latent parent variables.

