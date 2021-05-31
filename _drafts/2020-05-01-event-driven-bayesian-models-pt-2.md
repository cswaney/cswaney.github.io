---
layout: post
author: Colin Swaney
title: Event-Driven Bayesian Models (Pt. II)
date: 2020-05-1
categories: [research]
category: research
tags: [machine learning, high-frequency trading]
excerpt: "<p>A continuation of a introduction to an interesting class of time-series models. In the second part, I take a look at inference methods, with a particular emphasis on Bayesian inference via Gibbs sampling.</p>"
---

We left off by introducing a model that combines a network model with a temporal model of event arrivals called a Hawkes model. The intensity of the $$n$$-th node in this model at time $$t$$ is given by

$$ \lambda_n(t) = \lambda_n^{(0)} + \sum_{s_m < t} A_{c_m, n} \cdot W_{c_m, n} \cdot \theta_{c_m, n} \cdot e^{-\theta_{c_m, n} (t - s_m)}, $$

where $$A_{c_m, n} \in \{0, 1\}$$.

Simulating the model is fairly straight-forward using the Poisson Superposition Principle. First, simulate a network $$A$$ according to the generative model chosen (see previous lecture for details). Next, simulate independent events on each node according a to Poisson process with intensity given by the background rate, $$\lambda_{n}^{(0)}$$. Finally, for each event generated by the background process, recursively generate *independent* Poisson processes with intensity given by the impulse response functions.

## Inference
Now suppose we observe a sequence of events $$ \{ s_m, c_m \} $$, where $$c_m \in \{1, \cdots, N\}$$ represents the node of the event. How can we learn about the model parameters? Let's take a look at some of the standard approaches.

### Maximum Likelihood Estimation (MLE)
The classic approach to is to find the parameters $$ \{ \lambda^{(0)}, A, W, \theta \} $$ that maximize the likelihood of the data. One way to accomplish this is by introducing auxiliary parent variables $$\omega_m$$ denoting the event that "caused" the $$m$$-th event. The processes spawned by the events generated by each parent event are independent, so we can apply the Poisson Superposition Principle to compute the "augmented" likelihood:

$$ L( \{s_m, c_m, \omega_m\} \ | \ \lambda^{(0)}, A, W, \theta) = \prod_{n=1}^N \text{Poisson} \left( \lambda_n^{(0)} \right) \prod_{m=1}^M \text{Poisson} \left( h_{c_{\omega_m}, c_m} (s_m - s_{\omega_m}) \right)$$

To compute the likelihood, we sum over all possible values of each $$\omega_m$$:

$$ L( \{s_m, c_m, \omega_m\} \ | \ \lambda^{(0)}, A, W, \theta) = \prod_{n=1}^N \text{Poisson} \left( \lambda_n^{(0)} \right) \prod_{m=1}^M \text{Poisson} \left( \sum_{\ \omega_m < m} h_{c_{\omega_m}, c_m} (s_m - s_{\omega_m}) \right)$$

Unfortunately, this likelihood requires $$\mathcal{O}(M^2)$$ evaluations of the impulse-response function because *every* event prior to the $$m^{th}$$ is a potential parent (despite the fact that the probability decays exponentially). Nonetheless, we can evaluate the likelihood and maximize it numerically using standard methods. (Note as well that all of the model parameters are constrained to be *non-negative*).


### Maximum a Posteriori Estimation (MAP)
An alternative point estimation method is to calculate the parameters that maximize the posterior distribution. Recall that the posterior distribution of a probabilistic model with likelihood $$p(\mathcal{D} \ | \ \Theta)$$ and prior distribution $$p(\Theta \ | \ \nu)$$ is given by

$$ p(\Theta \ | \ \mathcal{D}, \nu) = \frac{p(\mathcal{D} \ | \ \Theta) p(\Theta \ | \ \nu)}{p(\mathcal{D})}, $$

where $$\mathcal{D}$$ represents the observed data, and $$\nu$$ represents the hyperparameters governing the prior distribution. The denominator is a constant with respect to $$\Theta$$. Thus, maximizing the posterior distribution is equivalent to maximizing the denominator. We have already seen how to calculate the likelihood, so let's introduce priors on the model parameters.

When the impulse-response rate is parameterized as an exponential function, it turns out that the conditional marginal distribution of each of model parameters is conjugate with a Gamma prior distribution:

$$ \lambda_n^{(0)} \sim \text{Gamma}\left(\alpha_0, \beta_0 \right)$$

$$ W_{n, m} \sim \text{Gamma}\left(\alpha_W, \beta_W \right)$$

$$ \theta_{n, m} \sim \text{Gamma}\left(\alpha_\theta, \beta_\theta \right) $$

The priors are independent, and the joint prior is therefore the product of the individual priors. With this information, we can again use standard numerical optimization methods to calculate $$\Theta_{MAP}$$.


### Bayesian Inference
MLE and MAP point estimates provide incomplete pictures of the model parameters. To fully explore the model, we want to sample from the marginal posterior distributions. We can use the Markov Chain Monte Carlo technique known as Gibbs sampling to do so. Gibbs sampling works by sampling from the conditional marginal distributions, which are often tractable, even though the joint distribution is intractable. Using gamma priors allows us to easily sample the conditional marginal distributions: they all follow a Gamma distribution due to conjugacy.

Let's walk through the priors one variable at a time. Start with the background intensity, $$\lambda^{(0)}$$. The marginal prior depends on the length of the observation period $$T$$, and the number of events that occur on each channel attributed to the background process,

$$M_{n}^{(0)} = \sum_{m=1}^M \mathbb{I}(c_m = n)\mathbb{I}(\omega_{c_m} = 0)$$

With these sufficient statistics, the prior is given by

$$ \lambda_n^{(0)} \sim \text{Gamma}(\alpha_0 + M_n^{(0)}, \beta_0 + T) $$

As data accumulates, the mean of this distribution converges to the average of $$M_n / T$$, or just the average rate of background events.

Similarly, for the weights we count up the number of events attributed to a given parent channel for each of the $$N$$ channels

$$ M_{n \rightarrow n'} = \sum_{m=1}^M \mathbb{I}(c_{\omega_{m}} = n) \mathbb{I}(c_m  = n'), $$

and the number of events on each channel,

$$ M_{n'} = \sum_{m=1}^M \mathbb{I}(c_m = n') $$

The posterior is then given by

$$ W_{n \rightarrow n'} \sim \text{Gamma}(\alpha_W + M_{n \rightarrow n'}, \beta_0 + M_{n'}) $$

In this case, as data accumulates, the mean of the prior approaches the average fraction of events on channel $$n'$$ attributed to a parent event on channel $$n$$.

Finally, the impulse-response likelihood is also conjugate with a gamma distribution prior. In this case, we need to calculate the average duration between parent and child events:

$$ X_{n \rightarrow n'} = \frac{1}{M_{n \rightarrow n'}} \sum_m \mathbb{I}(c_m = n') \mathbb{I}(c_{\omega_m} = n) s_m - s_{\omega_m} $$

The posterior is then

$$ θ_{n \rightarrow n'} \sim \text{Gamma}(\alpha_\theta + M_{n \rightarrow n'}, \beta_\theta + M_{n \rightarrow n'} \cdot X_{n \rightarrow n'}) $$

Note that the denominator of $$X_{n \rightarrow n'}$$ might be zero (if $$A_{n \rightarrow n'} = 0$$, for example). In that case, the we revert to the prior distribution. Intuitively, we haven't seen any data that can inform us about $$\theta{n \rightarrow n'}$$, so our best "information" continues to be our prior beliefs.

### Sampling the Network
The Hawkes process and the network model interface through the link matrix $$A$$. given $$A$$, we can perform Gibbs sampling on the Hawkes process as described above, and we can also perform Gibbs sampling on the network model parameters, as described below. But first, we need to sample $$A$$ itself.

We know that the posterior of $$A$$ is Bernoulli, and in fact it has the form

$$ p(A_{n \rightarrow n'} \ | \ \{s_m, c_m\}_{m=1}^M, \eta, \nu) = \frac{1}{Z} \times p(\{s_m, c_m\}_{m=1}^M \ | \ A, \eta) \times p(A_{n \rightarrow n'} \ | \ \nu), $$

where $$\eta = (\lambda^{0}, W, \theta)$$ and $$\nu$$ represents the parameters of the network model. This quantity is easy to calculate using the likelihood formula above that integrates over parent variables. The normalization constant, $$Z$$, is just the sum of the unnormalized "probabilities".

Given $$A$$, sampling the network model is straight-forward: $$A$$ is simply the "data" in the network model. (For example, in a Bernoulli network model, $$\rho$$ has an exact beta posterior distribution, $$ \rho \sim \text{Beta}(\alpha + N_0, \beta + N1)$$, where $$N_0$$ is the number of unlinked nodes and $$N_1$$ is the number of linked nodes).

### Computational Complexity
Inference via Gibbs sampling doesn't come cheap. The main bottleneck comes from sampling the auxiliary parent variables, which requires us to compute the intensity at *every* each previous event. Since the local variables are conditionally independent, we can *in principle* obtain an $$\mathcal{O}(M)$$ speed-up by sampling them in parallel. In practice, a more effective


## Examples
Let's look at some examples of these methods in action. We'll take a simple model containing just two nodes and a dense (fully-connected) network.

```julia
using Revise, Distributions, Gadfly, Distributed, PointProcesses

N = 2
λ0 = ones(N)
W = 0.1 * ones(N, N)
A = ones(N, N)
θ = ones(N, N)
α0 = 1.
β0 = 1.
κ = 1.
ν = ones(N, N)
αθ = 1.
βθ = 1.
net = DenseNetwork(N)
p = StandardHawkesProcess(λ0, W, A, θ, N, α0, β0, κ, ν, αθ, βθ, net)
T = 1000.;
events, nodes = rand(p, T);
```

### MLE


### MAP


### MCMC
In order to speed-up the Gibbs sampling algorithm, we take advantage of Julia's distributed computing capabilities (this still takes quite awhile to run, however).

```julia
using Distributed
addprocs(8)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses

M = 2000;
λ0, W, θ = mcmc(p, (events, nodes, T), M);
```

The resulting samples look alright (see below). It look like we could use a bit more data to tighten things up, but all the posterior distributions place a substantial amount of probability around the true parameter values.

![Gibbs Sampler: Background Rate](/assets/img/mcmc/background_mcmc.png)
![Gibbs Sampler: Weights](/assets/img/mcmc/weights_mcmc.png)
![Gibbs Sampler: Impulse Response](/assets/img/mcmc/impulse_response_mcmc.png)