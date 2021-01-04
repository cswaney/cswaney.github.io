---
layout: post
author: Colin Swaney
title: Event Models
date: 2020-04-25
categories: [research]
category: research
tags: [machine learning, forecasting]
excerpt: "<p>A summary of basic facts about point processes. I review continuous and discrete Poisson processes, including so-called Hawkes processes, and demonstrate several methods for estimating the parameters of these models from data using Julia.</p>"
---
<!-- I want to discuss a machine learning algorithm that transformed my understanding of what machine learning is, or rather what it *can be* in the hands of a strong mind. The class of models I want to discuss are rather complicated, but I hope to hold your hand through the process so that you can understand what is going on and hopefully illuminate to the model's brilliance and underlying simplicity.

To set the scene, imagine we wish to predict the occurrence of a stream of events. The events come from a variety of related sources or "channels". (The paper that developed this model was motivated by computational neuroscience research, so you can think of these channels as being neurons and the events as electric impulses being fired off). We want to quantify the probability of an event on each of these channels in real-time, but we would also like a model that is interpretable, allowing us to understand the structure of the system.


## Network Models


### Spike-and-Slab
So-called "spike-and-slab" models separate concerns between the existence of connections in a network and the strength of those connections. In particular, the network is represented by a random matrix of the form

$$ A \odot W, $$

where $$A$$ is a binary matrix and $$W \in \mathbb{R}^{N \times N}$$. $$A$$ represents the presence of a connection between nodes; $$W$$ captures the strength of the connection. To simplify, let's ignore $$W$$ for a moment and focus our attention on the connection matrix, $$A$$. How we chose to model $$A$$ reflects our understanding and beliefs about the nature of the system we are investigating. Consider the following three popular models:

#### Erdös-Rényi Network
In the Erdös-Rényi model, each connection in the model ($$a_{i, j}$$) is sampled independently from a \text{Bern} distribution:

$$a_{i, j} \sim \text{Bern}(\rho)$$

This modeling approach reflects a *lack* of structure in the network.

#### Stochastic Block Network
The Stochastic Block model adds (latent) structure to the network by assigning a class, $$z_i$$, to each node. Connections are sampled independently *conditional* on their class:

$$ a_{i, j} \sim \text{Bern}(\rho_{z_i}, \rho_{z_j}), $$

where $$z_k$$ is the latent class of the $$k$$-th node. The latent class itself requires a prior distribution, and a standard choice is a compound discrete distribution:

$$ z_i \sim \text{Discrete}(\pi), $$

$$ \pi \sim \text{Dir}(\alpha \mathbf{1}_K).$$

This approach is analogous to a Gaussian mixture model and reflects a belief that connections between particular nodes are more or less likely.

#### Latent Distance Network
If nodes are associated with characteristics/features, then it may be appropriate to model the likelihood of connections as dependent on the similarity between these features. In the Latent Distance Network, the distance between networks determines the probability of a connection between nodes. For a given metric, $$\| \cdot \|$$, the probability of a connection between nodes is

$$ p(a_{i, j} = 1 \ | \ z) = \sigma \left( - \| z_i -z_j \|_2^2 + \gamma_0 \right) $$

The features $$z$$ can be either observed features or latent class features. In the latter case, we need to specify a prior distribution of latent locations. A Normal-InverseGamma prior is a standard choice:

$$ z_i \sim \mathcal{N} (0, \tau I), $$

$$ \tau \sim \text{IGa}(1, 1).$$ -->




<!-- ## Outline

I. Poisson Processes
	a. Continuous
	b. Discrete
	c. Multivariate

II. Hawkes Process
	a. Continuous
	b. Discrete
	c. Multivariate

III. Inference
	a. Maximum-Likelihood Estimation
	b. Markov Chain Monte Carlo
	c. Variational Inference -->

## I. Poisson Processes

Continuous point processes model sequences of events $$\{s_m\}_{m=1}^M$$. Typically, $$s_m$$ denotes the time at which an event occurs, $$s_m \in \left[0, T\right]$$, but it could just as well represent the location where an event occurs, $$s_m \in \mathbb{R}^D$$. For now, we will consider the point processes as a univariate process that measures a single type of event; in general, a point process can represent multiple—possibly interacting—event streams.

A *Poisson process* is a point process in which the probability of events is determined by an intensity function $$\lambda(t)$$ such that the number of events that occur in a period $$\left[t, t + \Delta t \right]$$ has a Poisson distribution,

$$M \sim \text{Poisson}\left(\int_{t}^{t + \Delta t} \lambda(t) dt \right),$$

and the number of events in non-overlapping periods $$\mathcal{V}$$ and $$\mathcal{W}$$ are independent.


### Simulation
There is general method for simulating a Poisson process. It consists of sampling a number of events according to the Poisson distribution above, then sampling the time of these events according to the distribution of $$\lambda(t)$$ throuhout $$\left[0, T\right]$$,

$$ s_m \sim p(s) = \frac{\lambda(t)}{\int_0^T \lambda(t) dt}.$$

The first step ensures that the number of events reflects the aggregate intensity; the second step forces the timing of events to match the variation in intensity over time.


### Likelihood
The likelihood of a sequence of events generated according to a Poisson process is calculated in an analogous fashion to the simulation process. First, calculate the probability of observing $$M$$ events,

$$p(M) = \frac{\left( \int_0^T \lambda(t) dt \right)^{M}e^{-\int_0^T \lambda(t) dt}}{M!}.$$

Notice that this step ignores the *timing* of the events. Next, calculate the likelihood of the *collection* of event times $$\{s_m\}_{m=1}^M$$. The likelihood of an event at time $$s$$ is given by $$\frac{\lambda(s)}{\int_0^T \lambda(t) dt},$$ and therefore a *sequence* of events has likelihood

$$\prod_{m=1}^M \frac{\lambda(s_m)}{\int_0^T \lambda(t) dt}.$$

However, the events in $$\{s_m\}_{m=1}^M$$ are unordered. Thus, we multiply the above likelihood by $$M!$$ to account for all the possible ways of associating times with events. After simplifying, the overall likelihood is given by

$$L(\{s_m\} \ | \ \lambda) = \exp \left(-\int_0^T \lambda(t) dt \right) \prod_{m=1}^M \lambda(s_m)$$


#### Example: Homogeneous Poisson Process
Setting $$\lambda(t) = \lambda$$ gives us the simplest Poisson process, known as a *homogeneous* Poisson process. In any interval $$\left[t, t + \Delta t \right]$$, the number of events follows a Poisson distribution with intensity $$\lambda \Delta t$$. Thus, the mean and variance of the number of events in any period is proportional to the length of the period.

Below is an implementation of the homogeneous Poisson process in Julia. The struct is defined by a single parameter and has methods to simulate a sequence of events as well as calculate the likelihood of events.

```julia
using Distributions
import Base.rand

"""
A homogeneous Poisson process.
"""
struct HomogeneousProcess <: PoissonProcess
    λ
end

intensity(p::HomogeneousProcess) = t -> p.λ

function likelihood(p::HomogeneousProcess, ts, T)
    a = exp(-p.λ * T)
    b = p.λ ^ length(ts)
    return a * b
end

function rand(p::HomogeneousProcess, T)
    n = rand(Poisson(p.λ * T))
    ts = rand(Uniform(0, T), n)
    return sort(ts)
end
```


Note that for a homogeneous Poisson process, the second step of generating a random sample amounts to sampling from a uniform distribution on $$[0, T]$$. In addition, the likelihood of a homogeneous process simplifies to

$$ L(\{s_m\} \ | \ \lambda) = \exp \left(- \lambda T \right) \lambda^M $$ 

This likelihood is shownn below as a function of $$\lambda$$ for a sequence of ten events occuring over the period $$[0, 10]$$. For a homogeneous Poisson process, the timing of events is unimportant because they are uniformly distributed throughout time. Notice that the likelihood peaks at an intensity of one, which matches the average rate of events in the sample.

![homogeneous-poisson-process-likelihood](/assets/img/homogeneous-poisson-process-likelihood.svg)


## II. Hawkes Processes
Homogeneous Poisson processes wont get us too far. In interesting applications, the probability of events changes over time in response to environmental factors, perhaps dramatically so. We can model such environments through time-varying intensity functions augmented by environmental variables,

$$\lambda(t) = \lambda(t, x_t).$$

This definition says that the intensity can vary deterministically as a function of time or in response to a *state variable*, $$x_t$$. The relationship between $$\lambda$$ and $$x_t$$ can be nonlinear or even time-varying, and the state variable can therefore drive complicated dynamics. However, for some processes the intensity is better characterized as "self-exciting": when one event occurs, it increases the likelihood of additional events. Hawkes processes are a class of point processes with this property.

Specifically, the intensity of a Hawkes process is defined as

$$\lambda(t) = \lambda^{(0)} + \sum_{s_m < t} w \cdot \theta e^{-\theta (t - s_m)}.$$

According to this formulation, there is a jump in the intensity immediately following each event. The intensity then decays exponentially towards $$\lambda^{(0)}$$. If no events have occured recently, then we expect approximately $$\lambda^{(0)}$$ events to occur per unit of time, and we therefore refer to $$\lambda^{(0)}$$ as the *baseline intensity*[^1].

Notice that $$\theta e^{-\theta (t - s_m)}$$ is an exponential distribution and therefore integrates to one. As a result, each event occurrence is expected to directly generate an additional $$w$$ events. Thus, $$w$$ controls the amount of self-excitation, while $$\theta$$ controls its timing, with larger values of $$\theta$$ leading to slower decay.

[^1]: Note that we could also make the baseline intensity time-varying, $$\lambda^{(0)}(t)$$.

![univariate-hawkes-process-intensity](/assets/img/univariate-hawkes-process-intensity.png)


### Simulation
Simulation of Hawkes processes is complicated by their autoregressive nature. Previously, we were able to generate a random number of events using the fact that the number of events has a Poisson distribution with mean given by the integrated intensity of the process. In the case of a Hawkes process, the intensity depends on the number of events, so this approach is clearly infeasible.

However, there is an interesting property of Poisson properties that allows us to recover a straight-forward generative model. It turns out that there is no observational difference between a collection of independent Poisson processes and a single Poisson process whose intensity equals the sum of the intensities in the collection. Specifically, suppose a statisticain observe the events (but not the source) from $$K$$ *independent* Poisson processes with intensities $$\lambda_k$$. From the statistician's perspective, there is no differences between the $$K$$ processes and a single process with intensity

$$\lambda_{tot}(t) = \sum_{k=1}^K \lambda_k(t).$$

This property is known as the *Poisson Superposition Principle*, and in the case of a univariate Hawkes process it means that we can treat decompose a process into independent components, perform calculations, and then aggregate the results. In particular, every event is associated with either an independent Poisson process defined by its *impulse response*, $$w\theta e^{-\theta (t - s_m)},$$ or the baseline intensity, $$\lambda^{(0)}$$. Thus, simulating a Hawkes process amounts to simulating these component processes—which can be done using the previous two-step procedure—and then aggregating the results.

<!-- Figure. Hawkes simulation. -->


### Likelihood
Theoretically, the likelihood calculation for a Hawkes process is identical to the general formula for Poisson processes above. Assume for simplicity that $$\lambda^{(0)}$$ is constant. Then the likelihood is given by

$$L(\{s_m\} \ | \ \lambda) = \exp \left(-\int_0^T \lambda(t) dt \right) \prod_{m=1}^M \lambda(s_m)$$

$$= \exp \left(-\int_0^T \lambda^{(0)} dt - \sum_{m=1}^M \int_{s_m}^T h(t - s_m) dt \right) \prod_{m=1}^M \lambda^{(0)} + \sum_{m' < m} h(s_m - s_{m'})$$

Generally, we will work with the log-likelihood instead. Taking logs and simplifying the above we get

$$l(\{s_m\} \ | \ \lambda) = \lambda^{(0)}T + wM + \sum_{m=1}^M \log \left( \lambda^{(0)} + \sum_{m' < m} h(s_m - s_{m'}) \right)$$


In practice, several tricks prove useful in calculating the log-likelihood. First, the integrated intensity is broken down into independent components parts as in the simulation process. Standard integration methods can be used to determine the contribution of the baseline intensity and each of the impulse responses contributes $$w$$ to the total integrated intensity[^2].

[^2]: The contribution is slightly less than $$w$$ because the exponential kernel has infinite domain and events are measured over a finite time interval.

Second, notice that a naïve calculation of the intensity at event $$s_m$$ requires us to calculate an intensity for each event that occurred prior to $$s_m$$. Thus, the overall intensity calculation scales quadratically in the number of events, $$M$$. Later, we will see a algebraic trick to accelerate this calculation.

Finally, it is useful to recognize an alternative interpretation of the likelihood calculation above (this will help with the multivariate case below). Let us introduce an auxillary variable, $$\omega_m \in {0, 1, \dots, m-1}$$, which denotes the parent of the $$m$$-th event in the generative model. Since each of the impulse response functions represents an independent Poisson process, the likelihood of observing $$\{s_m \}_{m=1}^M$$ and $$\{\omega_m \}_{m=1}^M$$ is found by multiplying the likelihoods of the component processes, including the baseline process:

$$ \exp \left(-\int_0^T \lambda^{(0)} dt \right) \prod_{\omega_m = 0} \lambda^{(0)} \prod_{m=1}^M \exp \left( -\int_{s_m}^T h(t - s_m) dt \right) \prod_{\omega_{m'} = m} h(s_{m'} - s_m)$$

Now to get back the likelihood of the observed events, we sum over all possible parent combinations. Recalling that the sum of the products is the product of the sums[^3], this sum can be shown to equal

$$ \exp \left(-\int_0^T \lambda^{(0)} dt \right) \prod_{m=1}^M \exp \left( -\int_{s_m}^T h(t - s_m) dt \right) \prod_{m=1}^M \lambda{(0)} + \sum_{m' < m} h(s_m - s_{m'}).$$

You can verify that simplifying and taking logs gives the same formula as above for the log-likelihood. 

[^3]: That is, if $$x \in \{x_1, \dots, x_n\}$$ and $$y \in \{y_1, \dots, y_m\}$$, then the sum of all possible combinations of $$x$$ and $$y$$ is given by $$(x_1 + \cdots + x_n)(y_1 + \cdots + y_m)$$.

<!-- $$ L(\{s_m\}_{m=1}^M \ | \ \lambda_{tot}) = L(\{ \{s_m\}_{m=1}^{M_k} \}_{k=1}^K \ | \ \{ \lambda_k \}_{k=1}^K) $$ -->


#### Example: Hawkes Process with Homogeneous Baseline
The code below demonstrates an implementation of a univariate Hawkes process with a constant baseline intensity. Note that the likelihood calculation is written in a naïve (i.e. *slow*) fashion.

```julia
struct HawkesProcess
    λ0::Float64
    W::Float64
    θ::Float64
end

function intensity(p::HawkesProcess, events, t0)
    idx = events .< t0
    events = events[idx]
    λ = zeros(nchannels)
    ir = impulse_response(p)
    for childchannel = 1:nchannels
        for parenttime in events
            Δt = t0 - parenttime
            λ += ir(t0 - parenttime)
        end
    end
    return p.λ0 + λ
end

function impulse_response(p::HawkesProcess)
    P = ExponentialProcess(p.W, p.θ)
    return intensity.(P)
end

struct ExponentialProcess
    w
    θ  # rate  = 1 / scale
end

intensity(p::ExponentialProcess) = t -> p.w * pdf(Exponential(p.θ), t)

function loglikelihood(p::HawkesProcess, events, T)
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0) * T
    ll -= sum(p.W * length(events))
    # Calculate pointwise total intensity TODO: parallelize
    ir = impulse_response(p)
    for (childindex, childtime) in enumerate(events)
        λtot = p.λ0
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = 1
        while parentindex < childindex
            parenttime = events[parentindex]
            Δt = childtime - parenttime
            λtot += ir(Δt)
            parentindex += 1
        end
        ll += log(λtot)
    end
    return ll
end

function rand(p::HawkesProcess, T::Float64)
    events = Array{Float64,1}()

    # Generate events, starting from exogenous background processes
    # @info "generating exogenous events..."
    parent = HomogeneousProcess(p.λ0)
    childevents = rand(parent, T)
    # @info "> childevents = $childevents"
    append!(events, childevents)
    # generate endogenous events
    isempty(childevents) && continue
    parentevents = childevents
    for parentevent in parentevents
        # @info @sprintf("generating children for event %.2f...", parentevent)
        generate!(events, parentevent, p, T)
    end

    # return sorted events
    return sort(times)
end

function generate!(events, parentevent, process::StandardHawkesProcess, T)
    t0 = parentevent
    # @info "generating children..."
    parent = ExponentialProcess(process.w, process.θ)
    childevents = t0 .+ rand(parent, T - t0)
    # @info "childevents=$childevents"
    append!(events, childevents)
    isempty(childevents) && continue
    parentevents = childevents
    for parentevent in parentevents
        # @info @sprintf("generating children for event %.2f...", parentevent)
        generate!(events, parentevent, process, T)
    end
    # @info "done."
end
```


## III. Multivariate Processes
The concepts above extend to the case where we are interested in the dynamics of multiple time series. In the simplest case, we might model $$N$$ independent Poisson processes. In this case we can simply simulate indepednent processes and multiply likelihoods to obtain joint likelihoods. The situation is more complicated for multivariate Hawkes processes because the indiviudal series are dependent. But let us first define a multivariate Hawkes process.

A standard multivariate Hawkes process is specified as

$$ \lambda_n(t) = \lambda_n^{(0)} + \sum_{s_m < t} W_{c_m \rightarrow n} \cdot \theta_{c_m \rightarrow n} \cdot e^{-\theta_{c_m \rightarrow n} (t - s_m)} $$

The formula is similar to the univariate case except that we now have $$N$$ processes ($$n = 1, \dots, N$$) and each event amplifies the intensity of every process (in addition to itself). The effect of the $$n'$$-th process on the intensity of the $$n$$-th process is captured by $$W_{n' \rightarrow n}$$ and $$\theta_{n' \rightarrow n}$$. Clearly the processes are dependent, which complicates the calculation of likelihood, but allows us to model interactions between processes. It is convenient to think of this model as a representing a graph consisting of $$N$$ nodes with edges connecting each node in both directions. 

![Figure. A graph representation of a multivariate Hawkes process.](/assets/img/multivariate-hawkes-process-graph.png)


### III.A. Simulation
Simulation of a multivariate Hawkes process is handled using a similar approach to the univariate case based on the Poisson Superposition Principle. The only difference is that each event now spawns an independent Poisson process on *each* of the other $$N$$ channels.

![Figure. An example of the intensity of a multivariate Hawkes process.](/assets/img/multivariate-hawkes-process-intensity.png)

### III.B. Likelihood
Again, we can use the superposition principle to calculate likelihood. Recall that the principle requires *independent* processes. If we observed the "parent" of each event, $$\omega_m$$, then we could calculate the likelihood by computing $$\lambda_{tot}(s_m)$$ at each $$s_m$$ and apply the Poisson likelihood formula (noting that the integral of the exponential distribution is easy to compute). Unfortunately, the $$\omega_m$$ are latent variables in this model. A general strategy in this situtation is to compute the joint probability (i.e. including the latent parents, $$\omega_m$$) and then marginalizing over the parent variables to get back the likelihood.

This process works as follows. First, note that the probability that of event $$m$$ belonging to the process spawned by event $$m'$$ is given by

$$ p(\omega_m = m') = \frac{}{} $$


That is, by summing over all possible parents at each event (including the background process on the given node).
- Applying this to each of the $$N$$ processes, we get

$$ L(\{s_m, c_m\} \ | \ \theta, W) = \dots $$


<!-- ## Combining Network Models and Point Processes
To combine the spike-and-slab network model with the multivariate Hawkes process we simply modify the impulse-response to include the "spike" parameters, $$A_{i, j}$$.

$$ \lambda_n(t) = \lambda_n^{(0)} + \sum_{s_m < t} A_{c_m, n} \cdot W_{c_m, n} \cdot \theta_{c_m, n} \cdot e^{-\theta_{c_m, n} (t - s_m)}, $$

where $$A_{c_m, n} \in \{0, 1\}$$.

- The network model and the temporal model are *almost* disjoint: they connect through the influence of $$A$$ on the latent parent variables. -->


## IV. Inference

### Maximum-Likelihood Estimation (MLE)
### Maximum a Posteriori Estimation (MAP)
#### Markov Chain Monte Carlo (MCMC)


## V. Further Topics
- The intensity of a process can be parameterized as a function of exogenous factors:  $$\lambda(t) = \lambda(x_t, t)$$.
- We can also construct a model analogous to a hidden Markov model... (write down continuous-time Markov model for $$x_t$$)
- We can also construct a model analogous to a state-space model... (write down continuous-time autoregressive model for $$x_t$$)
- Do everything in discrete time in next lecture!