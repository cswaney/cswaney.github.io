---
layout: post
author: Colin Swaney
title: Variational Inference
date: 2020-06-1
categories: [research]
category: research
tags: [machine learning]
excerpt: "<p>A concise introduction to variational inference.</p>"
---

## Setting
- Assume a joint distribution of the form

$$ p(x, z, \beta \ | \ \alpha) = p(\beta \ | \ \alpha) \prod_{n=1}^N p(x_n, z_n \ | \ \beta) $$

- $$\beta$$ is a global variable with hyperparameter $$\alpha$$
- $$x = x_{1:N}$$ is the data/observations
- $$z = z_{1:N}$$ are local variables
- We call $$\{x_n, z_n\}$$ the $$n$$th "local context" (local in the sense that they are conditionally independent given the global variables, $$\beta$$).


- Assume that the "complete conditionals" are in the exponential family:

$$ p(\beta \ | \ x, z, \alpha) = h(\beta) \exp \{ \eta_g(x, z, \alpha)^{\intercal} t(\beta) - a_g(\eta_g(x, z, \alpha)) \} $$

$$ p(z \ | \ x, \beta) = h(z) \exp \{ \eta_l(x, \beta)^{\intercal} t(z) - a_l(\eta_l(x, \beta)) \} $$

- Assume that the prior distribution is also in the exponential family:

$$ p(\beta) = h(\beta) \exp \{ \alpha^\intercal t(\beta) - a_g(\alpha) \} $$

- These assumptions imply that the global variable has a complete conditional in the same exponential family as the prior with natural parameter

$$ \eta_g(x, z, \alpha) = (\alpha_1 + \sum_{n=1}^N t(z_n, x_n), \alpha_2 + N) $$

## Mean-Field Variational Inference
- Variational inference is based on converting the inference problem into an optimization problem. The value to be optimized—the "variational objective"—is called the "evidence lower bound", so-called because it describes a lower bound on the "evidence" $$p(x)$$.

- Suppose we introduce a distribution $$q(z, \beta)$$ that we think of as an approximation to $$p(z, \beta \ \vert \ x)$$. Using Jensen's inequality, we can show that

$$ \log p(x) \ge \mathbb{E}_q \left[ \log p(x, z, \beta) \right] - \mathbb{E}_q \left[ \log q(z, \beta) \right] = \text{ELBO} $$

- Alternative, consider the KL divergence between $$p(z, \beta \ \vert \ x)$$ and $$q(z, \beta)$$:

$$\text{KL}(q, p) = \mathbb{E}_q \left[ \log q(z, \beta) \right] - \mathbb{E}_q \left[ \log p(z, \beta \ \vert \ x) \right] = - \text{ELBO} + \text{const}.$$

- Minimizing the ELBO is thus equivalent to selecting the distribution $$q(z, \beta)$$ that is closest to the posterior.

- The "mean-field" family of variational distributions is a fully-factorized distribution

$$ q(z, \beta) = q(\beta \ \vert \ \lambda) \prod_{n=1}^N q(z_n \ \vert \ \phi_n) $$

- Moreover, we assume that the marginals belong to the same exponential family as the complete conditionals, with natural parameters equal to $$\lambda$$ and $$\phi_n$$ above:

$$ q(\beta \ \vert \ \lambda) = h(\beta) \exp \{ \lambda^{\intercal} t(\beta) - a_g(\lambda) \} $$

$$ q(z \ \vert \ \phi_n) = h(z_n) \exp \{ \phi_n^\intercal t(z_n) - a_l(\phi_n) \} $$

- Under these assumptions, we can calculate the ELBO as well as its gradient. It can be shown that:

$$ \nabla_\lambda \text{ELBO} = 0 \iff \lambda = \mathbb{E} \left[ \eta_g(x, z, \alpha) \right] $$

$$ \nabla_{\phi_n} \text{ELBO} = 0 \iff \phi_n = \mathbb{E} \left[ \eta_l(x, \beta) \right] $$

- This implies the following coordinate ascent algorithm maximizes the ELBO:


**NOTE**: The algorithm requires an update to **every** local variable before the global parameter is updated. This is quite wasteful.

## Stochastic Variational Inference

### Natural Gradients: a Detour
