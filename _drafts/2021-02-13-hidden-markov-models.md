---
layout: post
author: Colin Swaney
title: Hidden Markov Models
date: 2021-02-13
categories: [research]
category: research
tags: [machine learning]
excerpt: "<p></p>"
---
<!-- Add an introduction here. -->
Hidden Markov models are a classic way to model time series data. The basic idea is to combine two models together. The first model represents a hidden state of the world that moves along in a simple fashion. The second model describes what type of data we expect to observe conditional on the hidden state. Combining these two models together leads to more complicated time series than either model could achieve on its own. The most popular methods for estimating hidden Markov models is not Bayesian (that is, they don't calculate posterior distributions), but they rely on ideas that are unmistakenly Bayesian.

## Markov Chains
As the name suggests, hidden Markov models rely on something called a **Markov chain**. Markov chains are a simple way to model a sequence of discrete, finite values. The defining characteristics of Markov chains are:

1. The probability of moving from one state to another is always the same, and
2. The probability of moving from one state to another only depends on the current state.

A Markov chain is defined entirely by these probabilities, represented by a matrix, $$P$$, where the probability of transitioning from state $$i$$ to state $$j$$ is $$P_{i,j}$$.

What if we want to know the probability of transitioning from state $$i$$ to state $$j$$ in $$t$$ steps? Let's take $$t=2$$. If there are $$K$$ possible states, then there are $$K$$ different ways to arrive in state $$j$$: one way going through each of the possible states. To calculate the probability, we need to sum up the probability of each of these $$K$$ possible paths, which turns to be the $$P^2_{i,j}$$. It is easy to show that this calculation generalizes to any $$t$$. That is, the probability of moving the state $$i$$ to state $$j$$ in exactly $$t$$ steps is given by $$P^t_{i,j}$$. 

Markov chains have many interesting and useful properties, but we don't need to concern ourselves with those for now.

## Hidden Markov Models
<!-- Change the scenario to a coin toss -->
To understand the motivation behind hidden Markov models, let's start by considering a baseline model. Say that we observe a some discrete, finite quantity over time—repeated throws of a six-sided die, for example. We can represent this by a simple model: if $$x_t$$ represents the outcome of the $$t$$-th outcome, then the probability that $$x_t$$ is $$i$$ is given by $$p_i$$.

Now this is obviously not a very interesting model. The reason is that even though the observations take place over time, and we've bothered to index them by $$t$$, time plays no meaningful role. To make things more interesting, let's introduce a Markov chain. So suppose that we now have *two* six-sided die. One of these die is fair, so $$p_i = \frac{1}{6}$$ for all $$i$$. The other die has two had the six replaced by one such that $$p_1 = \frac{2}{6}$$ and $$p_6 = 0$$.

We generate a time series from these two die as follows. First, we choose one of the die to roll randomly. After we roll the die, we select the next die to roll according to Markov transition probabilities, $$P$$, and repeat. As long as we know which die is rolled each time period, this experiment is equivalent to two independent trials of the first experiment using two different die.

But what if we don't know which die is rolled?

In that case, we have arrived at an experiment that is exactly represented by a hidden Markov model. Hidden Markov models are more general than that, but the essential components are an unobserved Markov chain combined with an observed variable. Formally, a hidden Markov model consists of a **transition model**,

$$z_t | z_{t-1} \sim P_{i,j}$$

and an **observation model**,

$$x_t | z_t \sim P(x_t | z_t).$$

Typical choices for the observation model include the discrete model from the example above and the Gaussian model,

$$x_t | z_t = i \sim \mathcal{N}(\mu_i, \sigma^2_i).$$

<!-- ![HMM](/assets/img/hmm.pdf) -->

These types of models are clearly useful for representing situation where we believe there are important—but unobserved—forces at play. In such cases, the model can improve time series forcasts (because it is more accurate), but it also provides a method for infering the hidden state. In many situtations, the latter is of greater significance than the prior.

![Dishonest Casino](/assets/img/hmm/dishonest-casino.svg)

### Example: Boom-Bust Cycle
<!-- Example of expected return in high-low growth -->
As another example, consider the GDP growth of fictional country Kwazistan. Imagine that Kwazistan is subject to a pernicious "boom-bust" cycle. That is, it experiences periods of high growth, followed by periods or low growth, etc. We can model Kwazistan's monthly GDP growth as follows. Let $$z_t$$ represent economic conditions: $$z_t = 1$$ if the economy is booming, and $$z_t = 0$$ if Kwazistan is in a recession. Monthly GDP growth is represented by $$x_t$$ and is normally distributed conditional on the state of the economy:

$$x_t \sim \mathcal{N}(\mu_{z_t}, \sigma^2_{z_t}),$$

where $$\mu_1 > \mu0$$. If the state of the economy $$z_t$$ is a Markov chain, then this becomes a standard hidden Markov model with normal observations. The tools below provide a disciplined way of determining whether Kwazistan is in a recession or expansion.

<!-- ![Boom-Bust](/assets/img/hmm/boom-bust.png) -->


## Inference
Inference is the process of estimating the values of the hidden state variables, $${z_t}_{t=1}^T$$. There are many different ways to go about this, depending on the nature of the problem. The two main methods are *filtering* and *smoothing*. Simply put, filtering means to infer the distribution of $$z_t$$ conditional on information known up to time $$t$$, whereas smoothing means to determine the same distribution using all available information. Both methods rely of a process on Bayesian updating. Filtering forms a belief about the current period's state given yesterday's state, then updates that belief after observing today's observation. Smoothing uses the same updates, but then runs updating in reverse.

### Filtering
Filtering is performed via the so-called "forwards algorithm". 
```julia
"""
Run forwards (filtering) algorithm. Returns filtered probabilities (`a`) and log evidence (`z`).
"""
function forwards(x, A, B, π0)
    T = length(x)
    K = length(π0)
    a = Array{Float64,2}(undef, T, K)
    Z = Array{Float64,1}(undef, T)
    a[1, :], Z[1] = normalize(B[:, x[1]] .* (A * π0))
    for t = 2:T
        a[t, :], Z[t] = normalize(B[:, x[t]] .* (A * a[t - 1, :]))
    end
    ll = sum(log.(Z))
    return a, ll
end

function normalize(u)
    Z = sum(u)
    v = u ./ Z
    return v, Z
end
```

### Smoothing
Smoothing combines the forwards algorithm with a second "backwards algorithm" to incorporate future information into our estimates. This process can effectively be thought of as filtering in reverse: updating our beliefs about time $$t$$ based on the future. It's important to understand that **all** future information relevant to time $$t$$ is contained in $$z_{t+1}$$ in a hidden Markov model. Graphically, all information "flows" through $$z_{t+1}$$ on its way to $$z_t$$.

<!-- Backwards Algorithm -->
<!-- $$ $$ -->

<!-- Forwards-Backwards Algorithm (Smoothing) -->
```julia
"""
Run forwards-backwards (smoothing) algorithm. Returns the smoothed probabilities (`γ`).
"""
function forwards_backwards(x, A, B, π0)
    α, _ = forwards(x, A, B, π0)
    β = backwards(x, A, B)
    γ = α .* β
    return γ ./ sum(γ, dims=2)
end

function backwards(x, A, B)
    T = length(x)
    L = hmm.L
    b = Array{Float64,2}(undef, T, L)
    b[T, :] .= 1
    for i = 1:T - 1
        t = T - i
        b[t, :] = A * (B[:, x[t + 1]] .* b[t + 1])
    end
    return b
end
```

![Dishonest Casino Inference](/assets/img/hmm/dishonest-casino-inference.svg)

### Viterbi
Instead of figuring out the most like state at each time, we could instead determine the most likely *sequence* of events. These aren't the same thing (you can try think of a two-period counter-example). For hidden Markov models, this type of inference is called *Viterbi* filtering, and it results in paths of $${z_t}_{t=1}{T}$$ that are even smoother than those generated by the smoothing algorithm. 
<!-- Viterbi Algorithm -->
```julia
"""
Compute the most probable sequence of states conditional on `x`, i.e., argmax_z_1:T p(z_1:T | x_1:T).
"""
function viterbi(hmm, x, π0)
    T = length(x)
    K = length(π0)
    z = Array{Int64,1}(undef, T)
    logδ = Array{Float64,2}(undef, T, K)
    α = Array{Int64,2}(undef, T, K)
    A = transition(hmm)
    # forwards
    ϕ = observation(hmm, x[1])
    logδ[1, :] = log.(π0 .* ϕ)
    for t = 2:T
        ϕ = observation(hmm, x[t])
        # q = A .* (δ[t - 1, :] * ϕ')  - REPLACE w/ LOG (FASTER + NO UNDERFLOW)
        q = log.(A) .+ (logδ[t - 1, :] .+ log.(ϕ)')
        a = argmax(q, dims=1)
        logδ[t, :] = q[a]
        α[t, :] = map(x -> x.I[1] ,a)
    end
    # Backwards: z[t] = a[t + 1, z[t + 1]] for T - 1, ..., 1
    z[T] = argmax(logδ[T, :])
    for i = 1:(T - 1)
        t = T - i
        println("t=$t, z[t+1] = $(z[t + 1])")
        z[t] = α[t + 1, z[t + 1]]
    end
    return z, α, logδ
end
```

### Example: Sneaky Casino
### Example: Boom-Bust Cycle

## Learning
Inference assumes that we already know the parameters of the model. For example, is a discrete observation model, it assumes that we know the transition probabilities $$P$$ as well as the observation matrix $$B$$. In some settings, these might be reasonable assumptions. But, in general, we will not know these values, and will instead need to *learn* them from the data.

Learning for hidden Markov models is complicated by the fact that the likelihood of data depends on the hidden state variable. These variables are effectively additional parameters, and their number of scales with the number of observations. Therefore, learning amounts to large-scale optimization.

Fortunately, the structure of the hidden Markov model permits a clever approach to optimization known as expectation-maximization. The key to this approach is to recognize that *if* we know the parameters, then inference provides an exact solution to the optimal latent variables. Furthermore, given beliefs about the latent variables, we can easily maximize likelihood with respect to the parameters. Expectation-maximization works by alternating between these inference and maximization steps, which turns out to force parameter estimates towards their likelihood-maximizing values.

Formally,...

<!-- Baum-Welch Algorithm -->
```julia

```

### Example: Sneaky Casino


### Example: Boom-Bust Cycle


Hidden Markov models are the simplest for of dynamic probabilistic model that you might think of. There are all sorts of extensions and complications that we might add to the model, for example, adding a "forcing" variable to observations, or permitting autocorrelated observations. You might also think about allowing the hidden variable to have a continuous distribution. We'll look at all of these topics in future posts.