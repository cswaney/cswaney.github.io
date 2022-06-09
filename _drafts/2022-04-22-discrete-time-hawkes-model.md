---
layout: post
author: Colin Swaney
title: Discrete-time Hawkes Models
date: 2022-04-29
categories: [research]
category: research
tags: [research]
excerpt: "<p></p>"
---

## I. Model
The discrete-time Hawkes model operates similarly to the continuous-time version, but with some *potential* advantages in terms of inference. Just like the continuous case, we assume that each node has a (possibly inhomogeneous) baseline intensity $$\lambda_{t,n}^{(0)}$$, and that nodes are connected via weights $$W_{m,n}$$ and directed links $$A_{m,n}$$, where $$m$$ denotes the parent node and $$n$$ the child node. Similar to the continuous-case, every event causes a intensity ripple, or "impulse response". However, we now define this impulse response via a collection of basis functions.

$$
\begin{equation}

\end{equation}
$$

Each basis function is a proper distribution, and thus, a convex combination of the basis functions is also a proper (mixture) distribution. Therefore, we define the impulse response as

$$
\begin{equation}

\end{equation}
$$

Putting things altogether we come up with the following likelihood:

$$
\begin{equation}
    p(\mathbf{s} \ | \ ) = \dots
\end{equation}
$$

Defining the convolution between vectors $$x$$ and $$y$$ as $$\dots$$, we can re-write this expression as

$$
\begin{equation}
    p(\mathbf{s} \ | \ ) = \dots
\end{equation}
$$

It is convenient to pre-compute the convolution terms $$\dots$$, as they feature prominently in the inference methods for this model.


## II. Inference

### Gibbs Sampler
Similar to the continous-time settings, Gibbs sampling is simplified by introducing auxiliary parent variables. In this case, the parent variables represent the number of events at a given time on a specified node attributed to a particular parent node and basis functon. The auxiliary parent variables provide additional information that we can use to derive closed-form complete conditional formulas. Specifically, define $$\omega_{t, n}^{(m, b)}$$ as the number of events on the $$n$$-th node at time $$t$$ are attributed to the $$b$$-th basis function on the $$m$$-th node. Conditional on the auxiliary variables, the likelihood becomes

$$
\begin{multline}
    p(\{ \mathbf{s}, \boldsymbol \omega \} \ | \ \mathbf{A}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda^{(0)}) = \prod_{t=1}^T\prod_{n=1}^N \text{Poisson}(\omega_{t,n}^{(0)} \ | \ \lambda_n^{(0)} \Delta t) \prod_{t=1}^T\prod_{n=1}^N\prod_{m=1}^N\prod_{b=1}^B \text{Poisson}(\omega_{t,n}^{(m,b)} \ | \ \lambda_{t, n}^{(m, b)} \Delta t)
\end{multline}
$$

where

$$
\begin{equation}
    \lambda_{t, n}^{(m, b)} = a_{m,n} w_{m,n} \theta_{m,n}^{(b)} \sum_{d=1}^D s_{t-d, m} \phi_b^{(d)} \triangleq a_{m,n} w_{m,n} \theta_{n,m}^{(b)} \hat{s}_{t,m,b}
\end{equation}
$$

#### A. Auxillary Parents
Parent variables can be sampled in much the same fashion as in the continuous case. That is, events are randomly assigned parents based on the potential parents' contribution to overall intensity. Letting $$\boldsymbol \omega_{t,n} := (\omega_{t,n}^{(0)}, \omega_{t,n}^{1,1}, \dots, \omega_{t,n}^{N,B})$$ count the number of events at time $$t$$ on node $$n$$ attributed each possible parent we have

$$
\begin{equation}
    p(\boldsymbol \omega_{t,n}) = \text{Mult}(\boldsymbol \omega_{t,n} \ | \ \boldsymbol \mu_{t,n}, s_{t,n}),
\end{equation}
$$

where $$\boldsymbol \mu_{t,n} := (\mu_{t,n}^{(0)}, \mu_{t,n}^{1,1}, \dots, \mu_{t,n}^{N,B})$$ and

$$
\begin{equation}
    \mu_{t,n}^{m,b} = \frac{\lambda_{t,n}^{m,b}}{\lambda_{t,n}^{(0)} + \sum_{m', b'} \lambda_{t,n}^{m',b'}}
\end{equation}
$$

#### B. Background Intensity
$$
\begin{align}
    p(\lambda_n^{(0)} \ | \ \{ \mathbf{s}, \boldsymbol \omega \}, \mathbf{A}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda_{-n}^{(0)}, \alpha_\lambda, \beta_\lambda) & \propto p(\{ \mathbf{s}, \boldsymbol \omega \} \ | \ \lambda_n^{(0)}, \mathbf{A}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda_{-n}^{(0)}) p(\mathbf{A}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda_{-n}^{(0)} \ | \ \lambda_n^{(0)}) p(\lambda_n^{(0)} \ | \alpha_\lambda, \beta_\lambda) \nonumber \\
    & \propto p(\{ \mathbf{s}, \boldsymbol \omega \} \ | \ \lambda_n^{(0)}, \mathbf{A}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda_{-n}^{(0)}) p(\lambda_n^{(0)} \ | \alpha_\lambda, \beta_\lambda) \nonumber \\
    & \propto \prod_{t=1}^T \text{Poisson}(\omega_{t,n}^{(0)} \ | \ \lambda_n^{(0)} \Delta t) \  \text{Gamma}(\lambda_n^{(0)} \ | \ \alpha_\lambda, \beta_\lambda) \nonumber \\
    & \propto \prod_{t=1}^T {\Delta t}^{\omega_{t,n}^{(0)}} {(\lambda_n^{(0)})}^{\omega_{t,n}^{(0)}} \exp \Big \{ -\lambda_n^{(0)} \Delta t \Big \}  \ {\lambda_n^{(0)}}^{\alpha_\lambda - 1} \exp \Big \{ - \beta_\lambda \lambda_n^{(0)} \Big \} \nonumber \\
    & \propto {\lambda_n^{(0)}}^{\alpha_\lambda + \sum_{t=1}^T \omega_{t,n}^{(0)}- 1} \exp \Bigg \{ - \lambda_n^{(0)} \left(\beta_\lambda + \sum_{t=1}^T \Delta t \right) \Bigg \} \nonumber \\
    & \propto \text{Gamma} \left(\lambda_n^{(0)} \ | \ \alpha_\lambda + \sum_{t=1}^T \omega_{t,n}^{(0)}, \beta_\lambda + T \Delta t \right) 
\end{align}
$$


#### C. Impulse-Response
$$
\begin{align*}
    p(\theta_{m,n}\ | \ \{ \mathbf{s}, \boldsymbol \omega \}, \mathbf{W}, \mathbf{A}, \boldsymbol \theta_{\neg(m,n)}, \boldsymbol \lambda^{(0)}, \boldsymbol \gamma) & \propto p(\{ \mathbf{s}, \boldsymbol \omega \} | \ \mathbf{W}, \mathbf{A}, \boldsymbol \theta_{\neg(m,n)}, \boldsymbol \lambda^{(0)}, \theta_{m,n}) p(\theta_{m,n}\ | \ \boldsymbol \gamma) \nonumber \\
    & \propto \prod_{t=1}^T \prod_{b=1}^B \text{Poisson}\left(\omega_{t,n}^{(m,b)} \ | \ \hat{s}_{t, m, b} a_{m,n} w_{m,n} \theta_{m,n}^{(b)} \Delta t \right) \text{Dir}(\theta_{m,n} \ | \ \boldsymbol \gamma) \nonumber \\
    & \propto \prod_{b=1}^B {\theta_{m,n}^{(b)}}^{\gamma +\sum_{t=1}^T \omega_{t,n}^{(m,b)} - 1} \exp \Bigg \{ - \sum_{b=1}^B \sum_{t=1}^T \hat{s}_{t,m,b} a_{m,n} w_{m,n} \theta_{m,n}^{(b)} \Delta t \Bigg \} \nonumber \\
    & \propto \prod_{b=1}^B {\theta_{m,n}^{(b)}}^{\gamma +\sum_{t=1}^T \omega_{t,n}^{(m,b)} - 1} \exp \Bigg \{ - \sum_{b=1}^B a_{m,n} w_{m,n} \theta_{m,n}^{(b)} \underbrace{\sum_{t=1}^T \hat{s}_{t,m,b}}_{\approx N_m} \Bigg \} \nonumber \\
    & \propto \prod_{b=1}^B {\theta_{m,n}^{(b)}}^{\gamma +\sum_{t=1}^T \omega_{t,n}^{(m,b)} - 1} \exp \Bigg \{ -  a_{m,n} w_{m,n} N_m \underbrace{\sum_{b=1}^B \theta_{m,n}^{(b)}}_{=1} \Bigg \} \nonumber \\
    & \propto \text{Dir}\left(\theta_{m,n} \ | \ \boldsymbol \gamma +\sum_{t=1}^T \boldsymbol \omega_{t,n}^{(m)} \right)
\end{align*}
$$

#### D. Connection Weights
If $$a_{m,n} = 0$$, then $$w_{m,n} = 0$$. Otherwise,

$$
\begin{align*}
    p(w_{m,n} \ | \ \{ \mathbf{s}, \boldsymbol \omega \}, \mathbf{W}_{\neg(m,n)}, \mathbf{A}, \boldsymbol \lambda^{(0)}, \kappa, \nu) & \propto p(\{ \mathbf{s}, \boldsymbol \omega \} | \ \mathbf{W}_{\neg(m,n)}, \mathbf{A}, \boldsymbol \theta, \boldsymbol \lambda^{(0)}, w_{m,n}) p(w_{m,n} \ | \ \kappa, \nu) \nonumber \\
    & \propto \prod_{t=1}^T \prod_{b=1}^B \text{Poisson}\left(\omega_{t,n}^{(m,b)} \ | \ \hat{s}_{t, m, b} a_{m,n} w_{m,n} \theta_{m,n}^{(b)} \Delta t \right) \text{Gamma}(w_{m,n} \ | \ \kappa, \nu) \nonumber \\
    & \propto {w_{m,n}}^{\kappa + \sum_{t,b} \omega_{t,n}^{(m,b)} - 1} \exp \Bigg \{ -w_{m,n} \underbrace{\left(\sum_{t,b} \hat{s}_{t,m,b} + \nu \right)}_{\approx N_m + \nu} \Bigg \} \nonumber \\
    & \propto \text{Gamma}\left( w_{m,n} \ | \ \kappa + \sum_{t,b} \omega_{t,n}^{(m,b)}, \nu + \sum_{t} s_{t,m} \right)
\end{align*}
$$

#### E. Adjacency Matrix
We use the marginal likelihood in this case because the auxiliary variables could otherwise force connections to be one, i.e., if $$\omega_{t,n}^{(m, b)} > 1$$ for any $$t, b$$, then $$a_{m,n} = 1$$.

$$
\begin{align*}
    p(a_{m,n} = 1 \ | \ \mathbf{s}, \mathbf{A}_{\neg(m,n)}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda^{(0)}, \mathbf{z}) & \propto p(\mathbf{s} \ | \ \mathbf{A}_{\neg(m,n)}, \mathbf{W}, \boldsymbol \theta, \boldsymbol \lambda^{(0)}, a_{m,n}) p(a_{m,n} = 1 \ | \ \mathbf{z}) \nonumber \\
    & \propto \prod_{t=1}^T \text{Poisson}\left(s_{t,n} \ | \ \lambda_{t,n} \right) p(a_{m,n} = 1 \ | \ \mathbf{z}) \nonumber \\
    & \propto \prod_{t=1}^T \left( \lambda_{t,n} \Delta t \right)^{s_{t,n}} \exp \Bigg \{- \sum_{t=1}^T \lambda_{t,n} \Delta t \Bigg \} p(a_{m,n} = 1 \ | \ \mathbf{z}) \nonumber \\
    & \approx \prod_{t=1}^T \left( \lambda_{t,n} \Delta t \right)^{s_{t,n}} \exp \Bigg \{- \left( T \lambda_n^{(0)} + \sum_{k=1}^N a_{k,n} w_{k,n} N_k \right) \Delta t \Bigg \} p(a_{m,n} = 1 \ | \ \mathbf{z}) \nonumber \\ 
\end{align*}
$$

For numerical stability, taking logarithms gives the \textit{unnormalized} log probability

$$
\begin{align*}
    \log \tilde{p}_1= \sum_{t=1}^T s_{t,n} \log \lambda_{t,n} \cdot \Delta t - \Delta t \left(T \cdot \lambda_{n}^{(0)} +  \sum_{k=1}^N a_{k,n} w_{k,n} N_k \right) + \log p(a_{m,n} = 1 \ | \ \mathbf{z}) \nonumber \\
\end{align*}
$$

Note that $$a_{m,n}$$ is set to one in the summation. Similarly,

$$
\begin{align*}
    \log \tilde{p}_0= \sum_{t=1}^T s_{t,n} \log \lambda_{t,n} \cdot \Delta t - \Delta t \left(T \cdot \lambda_{n}^{(0)} +  \sum_{k=1}^N a_{k,n} w_{k,n} N_k \right) + \log p(a_{m,n} = 0 \ | \ \mathbf{z}), \nonumber \\
\end{align*}
$$

where $$a_{m,n}$$ is set to zero in the summation. We normalize these probabilites using the log-sum-exponent trick, e.g.,

$$
\begin{align*}
    p_1 = \frac{e^{\log \tilde{p}_1}}{e^{\log \tilde{p}_1} + e^{\log \tilde{p}_0}} = \frac{e^{\log \tilde{p}_1 - c}}{e^{\log \tilde{p}_1 - c} + e^{\log \tilde{p}_0 - c}},
\end{align*}
$$

where $$c = \max \{ \log \tilde{p}_1, \log \tilde{p}_0 \}$$.

#### F. Network

### Variational Inference

#### A.
#### B.
#### C.
#### D.