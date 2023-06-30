---
layout: post
author: Colin Swaney
title: Variational Inference
date: 2020-06-1
categories: [research]
category: research
tags: [machine learning]
excerpt: "<p></p>"
---

Consider a general probabilistic model of the form

$$ p(x, z, \beta \ | \ \alpha) = p(\beta \ | \ \alpha) \prod_{n=1}^N p(x_n, z_n \ | \ \beta) $$

In this joint distribution, $$\beta$$ is a global variable (determined by the hyperparameter $$\alpha$$), $$x_{1:N}$$ is observed data, and $$z_{1:N}$$ represents local latent variables. We refer to $$\{x_i, z_i\}$$ as the $$i$$th "local context". It is local in the sense that it is conditionally independent given the global variable.

We will assume that the "complete conditionals" are in the exponential family,

$$ p(\beta \ | \ x, z, \alpha) = h(\beta) \exp \{ \eta_g(x, z, \alpha)^{\intercal} t(\beta) - a_g(\eta_g(x, z, \alpha)) \} $$

$$ p(z \ | \ x, \beta) = h(z) \exp \{ \eta_l(x, \beta)^{\intercal} t(z) - a_l(\eta_l(x, \beta)) \}, $$

as well as the prior distribution,

$$ p(\beta) = h(\beta) \exp \{ \alpha^\intercal t(\beta) - a_g(\alpha) \} $$

These are mild restriction: given the latter, the prior is satisfied by any conjugate model. In fact, the assumptions imply that the global variable has a complete conditional in the same exponential family as the prior with natural parameter

$$ \eta_g(x, z, \alpha) = (\alpha_1 + \sum_{n=1}^N t(z_n, x_n), \alpha_2 + N) $$

## Mean-Field Variational Inference
Variational inference works by converting the inference problem (i.e. determining the posterior distribution) into an optimization problem. The value to be optimized—the "variational objective"—is called the "evidence lower bound", so-called because it provides a lower bound on the "evidence", $$p(x)$$, provided an approximation of the posterior distribution, $$q(z, \beta)$$.

We think of $$q(z, \beta)$$ that we think of as an approximation to $$p(z, \beta \ \vert \ x)$$, which is referred to as the "proposal" distribution. Using Jensen's inequality, we can show that

$$ \log p(x) \ge \mathbb{E}_q \left[ \log p(x, z, \beta) \right] - \mathbb{E}_q \left[ \log q(z, \beta) \right] = \text{ELBO} $$

The righthand side of the above equation is the evidence lower bound, or ELBO. Maximizes the ELBO is equivalent to finding the proposal distribution that is "closest" to the true posterior distribution. Distance between probability distributions is often measured by their KL divergence. For distributions $$p(z, \beta \ \vert \ x)$$ and $$q(z, \beta)$$ the KL divergence is

$$\text{KL}(q, p) = \mathbb{E}_q \left[ \log q(z, \beta) \right] - \mathbb{E}_q \left[ \log p(z, \beta \ \vert \ x) \right] = - \text{ELBO} + \text{const},$$

where the constant represents terms that don't depend on $$q$$.

Mean-field variational inference relies a family of variational distributions that is fully-factorized

$$ q(z, \beta) = q(\beta \ \vert \ \lambda) \prod_{n=1}^N q(z_n \ \vert \ \phi_n) $$

Moreover, we assume that the marginal distributions belong to the same exponential family as the complete conditionals above:

$$ q(\beta \ \vert \ \lambda) = h(\beta) \exp \{ \lambda^{\intercal} t(\beta) - a_g(\lambda) \}, $$

$$ q(z \ \vert \ \phi_n) = h(z_n) \exp \{ \phi_n^\intercal t(z_n) - a_l(\phi_n) \}, $$

with natural parameters $$\lambda$$ and $$\phi_n$$.

Under these assumptions, we can calculate the ELBO as well as its gradient. In fact, the first-order conditions take a particularly simple form:

$$ \nabla_\lambda \text{ELBO} = 0 \iff \lambda = \mathbb{E} \left[ \eta_g(x, z, \alpha) \right] $$

$$ \nabla_{\phi_n} \text{ELBO} = 0 \iff \phi_n = \mathbb{E} \left[ \eta_l(x, \beta) \right] $$

This implies the following coordinate ascent algorithm:

##### Algorithm: (Mean-Field) Variational Inference
>
- Select a random $$\lambda^{(0)}$$
- Repeat until convergence:
  - Update each local variational parameter, $$ \phi_i^{(t)} = \mathbb{E}_{\lambda^{(t - 1)}} \left[ \eta_l (x_{1:N}, \beta) \right] $$
  - Update the global variational parameter, $$ \lambda^{(t)} = \mathbb{E}_{\phi^{(t)}} \left[ \eta_g (x_{1:N}, z_{1:N}) \right] $$

In practice, the updates would be converted back to their usual representation/parameterization (instead of working directly with the natural parameters—see the example below).


## Stochastic Variational Inference
The variational inference algorithm above requires an update to *every* local variable before the global parameter is updated. This is inefficient because the expectation of the global natural parameter is the sum over independent local contexts (observation plus local latent variables), and we can therefore approximate the update in an unbiased fashion using any random subsample.

Specifically, in the case of mean-field variational inference, the global natural parameter (ignoring the second component) is

$$ \eta_g(x, z, \alpha) = \alpha + \sum_{n=1}^N t(x_n, z_n) $$

Replacing $$ \sum_{n=1}^N t(x_n, z_n) $$ by $$ N \times t(x_i, z_i) $$ for a randomly chosen $$i \in 1, \dots, N$$ results in an unbiased estimate of the true sum, and is obviously *much* cheaper than using all of the samples when $$N$$ is large. The new overall stochastic variational inference algorithm therefore looks like this:

##### Algorithm: (Mean-Field) Stochastic Variational Inference
>
- Select a random $$\lambda^{(0)}$$
- Repeat until convergence:
  - Select a random data point, $$ x_i \sim \text{Uniform}(\{x_i\}_{i=1}^N) $$
  - Update the local variational parameter, $$ \phi = \mathbb{E}_{\lambda^{(t - 1)}} \left[ \eta_l (x, \beta) \right] $$
  - Update the global variational parameter, $$ \lambda^{(t)} = \mathbb{E}_{\phi} \left[ \alpha + N \times t(x_i, z_i) \right] $$

This is the only real difference between stochastic variational inference and plain-ole variational inference. But nonlinear optimization, and stochastic optimization in particular, contains many tricks and variations that can improve the stability of speed of convergence depending on the nature of the problem at hand, and certainly you can expect to improve on this "vanila" algorithm by adopting some of those strategies.

For example, in Hoffman et al. (2013), the authors actually recommend an algorithm based on the "natural gradient", an idea that shows up in reinforcement learning and deserves its own lecture. Simply put, the natural gradient accounts for the topology/geometry of the optimization problem introduced by the probability distribution, which tends to lead to better results. (Remember, the real problem that we're trying to solve is finding the distribution in one family that is closest to a distribution in some other family. The action is happening in an abstract space).

There is nothing magical going on here: this is "just" a natural modification of the coordinate ascent algorithm, which is pervasive in machine learning algorithms (most notably, deep learning). It is an rather important modification because it allows the algorithm to *scale*. We want to perform inference on complex models in domains with abundant datasets, and the nature of these models is such that the number of variables scales with the size of the data. This simple extension means that the cost of our inference algorithm grows at a substantially lower rate. One way to think about this is that we don't really need to see every observation in a dataset to understand what is going on at a global level. Once we have seen "enough" data, we know how the model works—the additional data is gratuitous. (On the other hand, if the model is complex, then "enough" data might be quite a bit).

By the way, this is why you need to mark independence in the Pyro probabilistic programming language if you want to get reasonable performance: it lets the package know where it can use stochastic optimization. If you don't mark independence, the algorithm will still work, but it will essentially fall back to treating everything as a global variable. You can think of the process as marking local context.

Finally, you might wonder if using stochastic gradient optimization is really an advantage. After all, the approximation makes the updates noisier. There are a few reasons why this tends to work better in practice. For one thing, remember that we are starting from a random guess, which means that there isn't much reliable information in the early stages of optimization: taking a full step means acting with certainty on poor data. Adding randomness might also be beneficial by preventing the algorithm from getting stuck in a *local* optimum.


## Example: Word Model
This example is a simplification of the application in Hoffman et al. (2013). Here we imagine a single document in which each word has a specific type, and the probability of each word depends on its type. For example, if the type of the word is "sports noun", then the distribution might place higher probability on words like "ball" and "throw". There are a finite number of word types and a finite number of words. The model looks like this:

$$ \theta_z \sim \text{Dirichlet}(\alpha_z) $$

$$ \theta_{w_k} \sim \text{Dirichlet}(\alpha_w) \ \ \ (k = 1, \dots, K)$$

$$ z_i \sim \text{Cat}(\theta_z) $$

$$ w_i \sim \text{Cat}(\theta_{w_{z_i}}) $$

The model is fully-conjugate, thus the variational distributions are

$$ \theta_z \sim \text{Dirichlet}(\tilde{\alpha}_z),$$

$$ \theta_{w_k} \sim \text{Dirichlet}(\tilde{\alpha}_w),$$

and

$$ z_i \sim \text{Cat}(\tilde{\theta})  $$

where $$\tilde{\alpha}_z$$, $$\tilde{\alpha}_w$$, and $$\tilde{\theta}$$ are variational parameters.

The complete conditionals of this model are

$$ \tilde{\alpha}_k = \alpha_k + \sum_{i=1}^N \mathbb{I}(z_i = k) $$

$$ \theta \ \vert \ \{w_i, z_i \}_{i=1}^N, \{ \theta_{w_k} \}_{k=1}^K \sim \text{Dirichlet}(\tilde{\alpha}) $$

$$ \tilde{\gamma}_j^{(k)} = \gamma_j^{(k)} + \sum_{i=1}^N \mathbb{I}(z_i = k) \mathbb{I}(w_i = j) $$

$$ \theta_w^{(k)} \ \vert \ \{w_i, z_i \}_{i=1}^N, \theta_z \sim \text{Dirichlet}(\tilde{\gamma}^{(k)}) $$

$$ \tilde{\theta}_k \propto \theta_z^{(k)} \times p(w_i \ \vert \ z_i = k) $$

$$ z_i \ \vert \ w_{1:N}, z_{-i}, \theta_z, \theta_w \sim \text{Cat}(\tilde{\theta}) $$
