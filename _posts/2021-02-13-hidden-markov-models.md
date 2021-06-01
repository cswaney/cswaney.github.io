---
layout: post
author: Colin Swaney
title: Hidden Markov Models
date: 2021-02-13
categories: [research]
category: research
tags: [machine learning]
excerpt: "<p>A brief summary of hidden Markov models, including an explanation of methods for inference and estimation.</p>"
---
Hidden Markov models are a classic approach to represent time series data using a combination of models. One model represents a hidden state of the world that bounces along in a simple, random fashion. The second model describes the data we observe conditional on the hidden state. Joining these two models permits more complex behavior than either model can achieve on its own.

These notes summarize methods commonly used for inference and estimation of hidden Markov models. Popular methods are not Bayesian per se (they do not calculate posterior distributions), but they rely on unmistakenly Bayesian ideas. We will closely follow the presentation in [Murphy, 2012], and the interested reader can refer there for additional detail and references.


## Markov Chains
As the name suggests, hidden Markov models build upon *Markov chains*, which provide a base for time series analysis. Markov chains model sequences of discrete states such that:

1. The probability of moving from one state to another is always the same, and
2. The probability of moving from one state to another only depends on the current state.

A Markov chain is defined entirely by its transition probabilities, represented by a matrix, $$\mathbf{A}$$, where $$a_{i,j} = \mathbf{A}\left[i,j\right]$$ denotes the probability of transitioning from state $$i$$ to state $$j$$.

What if we want to know the probability of transitioning from state $$i$$ to $$j$$ in exactly $$t$$ steps? Let's begin with the case where $$t=2$$. If there are $$k$$ possible states, then there are $$k$$ different ways to arrive at state $$j$$: one way going through each of the possible states. To calculate the probability, we need to sum up the probabilities of the possible paths, which is conveniently calculated as $$\mathbf{A}^2$$. In fact, this calculation generalizes to any $$t$$: the probability of moving from state $$i$$ to state $$j$$ in exactly $$t$$ steps is given by $$\mathbf{A}^t$$. 

Markov chains possess many more interesting and useful properties, but these are the essential facts we require for now.


## Hidden Markov Models
To understand the motivation behind hidden Markov models, let's start by looking a baseline model. Suppose that we observe some discrete, finite quantity over time—repeated tosses of a coin, for example. We can represent this with a simple model: if $$x_t$$ represents the outcome of the $$t$$-th toss, then the probability that $$x_t = i$$ is given by $$p_i$$.

Now this is obviously not a terribly interesting model. Although the observations take place over time (and we've bothered to index them by time), time plays no meaningful role. To make things more interesting, let's introduce a Markov chain. Suppose that we now have two coins. One of these coins is fair, while the other is modified such that $$p_1 = \frac{4}{5}$$ and $$p_2 = \frac{1}{5}$$.

We generate observations from these two coins as follows. First, we randomly choose one of the coins to toss. After we toss the coin, we select the next coin to toss according to Markov transition probabilities, $$P$$, and repeat. If we know which coin is tossed each period, this experiment is equivalent to two independent trials of the first experiment using two different coins. But what if we don't know which coin is tossed?

In that case, our experiment is exactly represented by a hidden Markov model. Hidden Markov models can represent considerably more complex phenomena, but the experiment contains the essential components: an *unobserved* Markov chain and an *observed* time series.

Formally, a hidden Markov model consists of a *transition model*,

$$z_t \ \vert \ z_{t-1} \sim \mathbf{A}$$

and an *observation model*,

$$x_t \ \vert \ z_t \sim p(x_t \ \vert \ z_t).$$

Typical choices for the observation model include the discrete model from the example above and the Gaussian model,

$$x_t \ \vert \ z_t \sim \mathcal{N}(\mathbf{\mu}_{z_t}, \mathbf{\Sigma}_{z_t}).$$

Such models are clearly helpful for representing situations where we believe that unobserved forces play a significant role. In such cases, these models can improve time series forecasts, but they also provide a means of inferring hidden states of the world—an essential part of many scientific inquiries.

As another example, consider the GDP growth of a fictional country, Kwazistan, subject to a pernicious "boom-bust" growth cycle. Kwazistan experiences periods of expansive growth, followed by periods of low investment, high unemployment, and generally poor economic conditions. We might model Kwazistan's monthly GDP growth as follows. Let $$z_t$$ represent economic conditions: $$z_t = 1$$ if the economy is booming, and $$z_t = 0$$ if Kwazistan is in a recession. Monthly GDP growth is represented by $$x_t$$ and is normally distributed conditional on the state of the economy:

$$x_t \ \vert \ z_t \sim \mathcal{N}(\mu_{z_t}, \sigma^2_{z_t}),$$

where $$\mu_1 > \mu_0$$. If the state of the economy is a Markov chain, then this is a standard hidden Markov model with normally distributed observations. The methods described below will allow us to estimate whether Kwazistan is (or was) in a recession or expansion at any point in time.


## Inference
Inference is the process of estimating the values of hidden state variables, $$\{z_t\}_{t=1}^T$$. There are several ways to go about this, depending on the nature of the problem, but let us start with the basics: *filtering* and *smoothing*. Simply put, filtering infers the distribution of $$z_t$$ conditional on information known up to time $$t$$, whereas smoothing determines this distribution using all available information. Both methods rely on a process of Bayesian updating. Filtering forms a belief about the current period's state given yesterday's state, then updates that belief after observing today's data. Smoothing follows essentially the same updating procedure but then repeats the procedure in reverse, thereby incorporating future knowledge into estimates.


### Filtering
Our task is to determine beliefs about the current state, $$z_t$$, given all observations up until the current time and our beliefs about the previous state, $$z_{t-1}$$. We will break the problem into two parts. First, we translate beliefs about the previous state to beliefs about the current state *ignoring* the new information provided by the current observation. Next, we update the translated beliefs to incorporate the new observation. Let's denote our beliefs about state by vector $$a^{(t)}$$ such that $$a^{(t)}_i = p(z_t = i \ \vert \ x_t, \dots, x_1)$$. Given the Markov nature of the state variable, we see that for a transition matrix $$\mathbf{A}$$,

$$p(z_t = i \ \vert \ z_{t-1}) = \mathbf{A} a^{(t-1)}$$

We treat this probability as a prior and apply Bayesian updating:

$$a^{(t)} \propto \psi^{(t)} \odot \mathbf{A} a^{(t-1)},$$

where $$\psi^{(t)}$$ is a vector whose $$i$$-th element equals $$p(x_t \ \vert \ z_t = i)$$.

The figure below demonstrates the filter algorithm applied to the coin-tossing problem described above. In this example, the probability of remaining in the same state is high—90%—and design the to be easily distinguishable: the first is heads 90% of the time, while the second is tails with 90% probability. The blue dots represent the maximum-likelihood states, according to the filtered probabilities, $$a_t$$. This problem is quite simple for the filtering algorithm, as the filtered estimates (blue) closely follow the actual states (red).

![Filtering](/assets/img/hmm/filter.svg)
<div class="uk-text-center"><b>The Filtering Algorithm</b> applied to the coin-tossing problem. Red dots represent true states and blue dots represent filtered states, with y-values jittered for easier visualization.</div>


### Smoothing
In smoothing, our task is to estimate the current state given *all* observations, $$\mathbf{x}_{1:T} = x_1, \dots, x_T$$. The filter algorithm above tells us how to use all information prior to a given period. The same algorithm, run in reverse, estimates the hidden states using all information *following* a given period. In fact, we can solve our problem by simply following the filtering algorithm twice: once forwards and once backward [^smoothing].

However, we can also find a convenient formulation using the following decomposition:

$$p(z_t = j \ \vert \ \mathbf{x}_{1:T}) \propto p(z_t = j, \mathbf{x}_{t+1:T} \ \vert \ \mathbf{x}_{1:t}) \propto p(z_t = j \ \vert \ \mathbf{x}_{1:t}) p(\mathbf{x}_{t+1:T} \ \vert \ z_t = j),$$

where we've used the fact that $$\mathbf{x}_{t+1:T}$$ is independent of $$\mathbf{x}_{1:t}$$ conditional on $$z_t$$ in the final expression. Defining $$b_j^{(t)} = p(\mathbf{x}_{t+1:T} \ \vert \ z_t = j)$$ and the smoothed marginals by $$\gamma_j^{(t)} = p(z_t \ \vert \ \mathbf{x}_{1:T})$$, this gives the convenient expression

$$\gamma_j^{(t)} \propto a_j^{(t)} b_j^{(t)}$$

This expression is just Bayes rule: it states that our beliefs about $$z_t$$ are proportional to our "prior" beliefs (having seen data up to time $$t$$) times the likelihood of the (new) data. And it turns out that these likelihoods can be easily calculated recursively. Starting from

$$b_j^{(T)} = p(\mathbf{x}_{T+1:T} \ \vert \ z_T = j) = 1$$

we have

$$b_i^{(t-1)} = \sum_{j=1}^K p(z_t = j \ \vert \ z_{t-1} = i) p(x_t \ \vert \ z_t = j) p(\mathbf{x}_{t+1:T} \ \vert \ z_t = j),$$

Or, more compactly,

$$b_i^{(t-1)} = \sum_{j=1}^K \mathbf{A}_{i,j} \psi_j^{(t)} b_j^{(t)}$$

To sum up, smoothing requires two passes through the data—one forwards, one backward—each consisting of recursively multiplying by transition and observation probabilities. Given pre-computed observation probabilities, the same algorithm works for any observation model.

[^smoothing]: Moving backwards in time requires some modification to the algorithm, but works essentially the same.

The figure below demonstrates the smoothing algorithm applied to the coin-tossing example. It appears similar to the figure demonstrating the filtering algorithm on the same simulated data set. However, notice that around the period $$t=35$$, the state "unexpectedly" changes. The filtered estimates follow the state, but the smoothed estimates remain constant, demonstrating the source of the smoothing algorithm's name: it generates estimates that are "smoother" than those of the filtering algorithm. In this case, we can understand why this happens by looking at the data following the change in state, which quickly reverts to the first coin and remains there for the following 30 periods. *Ex-post* a change of state appears extremely unlikely, but this information isn't known to the filtering algorithm[^smoothed-vs-filtered].

[^smoothed-vs-filtered]: In this case, the filtering algorithm is correct, but obviously the smoothing algorithm gives superior estimates, on average, as it has access to a superior information set.

![Smoothing](/assets/img/hmm/smooth.svg)
<div class="uk-text-center"><b>The Smoothing Algorithm</b> applied to the coin-tossing problem. Red dots represent true states and blue dots represent smoothed states, with y-values jittered for easier visualization.</div>

It is worthwhile to make a brief observation about the conditional probability of $$z_t$$. When we condition on a sequence of observations, information passes forwards to our current location. That is, each $$x_1, \dots, x_{t-1}$$ contains information relevant to $$z_t$$. However, as soon as we resolve uncertainty at any prior hidden state, the observations up to that period become irrelevant. Intuitively, those observations only inform the current state insofar as they affect the now-revealed hidden state. The same is true in reverse, and we can summarize this observation by stating that

>> "All information flows through the states."

This property is a simple example of a more general property of probabilistic graphical models.


### Viterbi
Instead of figuring out the most likely state at each point in time, we could determine the most likely *sequence* of states. States are conditionally dependent (due to the Markov dynamics), and therefore a sequence that maximizes likelihood at each period does not generally maximize overall likelihood. For hidden Markov models, we call this type of inference *Viterbi* filtering, and it results in paths $$\{ z_t \}_{t=1}^{T}$$ that are more "internally consistent" than those generated by the smoothing algorithm.

Formally, our task is to find

$$ \mathbf{z}^* = \underset{\mathbf{z}_{1:t}}{\operatorname{arg max}} p(\mathbf{z}_{1:t} \ \vert \mathbf{x}_{1:T}) $$

Again, a recursive strategy presents itself. Let

$$ \delta_j^{(t)} = \max_{\mathbf{z}_{1:t-1}} p(\mathbf{z}_{1:t-1}, z_t = j \ \vert \ \mathbf{x}_{1:t}) $$

That is, $$\delta_j^{(t)}$$ is the (conditional) probability of arriving at state $$j$$ following a most likely path. If we follow the most likely path up to $$t-1$$, then we can continue on a most likely path at time $$t$$ by maximizing probability over the next step. This insight leads to the formula

$$ \delta_j^{(t)} \propto \max_i \delta_i^{(t-1)} p(z_{t -1} = i, z_t = j) p(x_t \ \vert \ z_t = j) $$

Furthermore, at each state, it allows us to find the the preceeding state along the most likely path:

$$ \hat{z}_j^{(t)} = \underset{i}{\operatorname{arg max}} \delta_i^{(t-1)} p(z_{t -1} = i, z_t = j) p(x_t \ \vert \ z_t = j) $$

Following this recursion to time $$T$$, we can determine the terminal state along the most likely path,

$$z_T^* = \underset{i}{\operatorname{arg max}} \delta_i^{(T)},$$

after which finding the full path is a simple matter of tracing back using the previous computed predecessors:

$$ z_{t}^* = \hat{z}_{z_{t+1}^*}^{(t+1)}$$

The only detail left is to determine where to start, but as there is no initial transition, we define

$$ \delta_j^{(1)} = p(z_1 = j \ \vert \ x_1) \propto p(z_1 = j)p(x_1 \ \vert \ z_1 = j).$$

Finally, note that in practice we can run the recursion on $$\log \delta^{(t)}$$ to avoid numerical underflow (which works because $$\log \max = \max \log$$).

![Viterbi Coding](/assets/img/hmm/viterbi.svg)
<div class="uk-text-center"><b>The Viterbi Algorithm</b> applied to the coin-tossing problem. Red dots represent true states and blue dots represent the maximum-likelihood states generated by the Viterbi coding, with y-values jittered for easier visualization.</div>


### Sampling
Another interesting idea is to generate sample paths conditional on the observed data, $$ \mathbf{z}_{1:T} \sim p(\mathbf{z}_{1:T} \ \vert \ \mathbf{x}_{1:T})$$. A straight-forward method to generate such samples involves running the forward-backward algorithm to calculate smoothed two-period probabilities, $$p(z_t, z_{t+1} \ \vert \ \mathbf{x}_{1:T})$$, and then calculating corresponding smoothed transition probabilities

$$p(z_t \ \vert \ z_{t-1}, \mathbf{x}_{1:T}) = \frac{p(z_{t-1}, z_t \ \vert \ \mathbf{x}_{1:T})}{p(z_{t-1} \ \vert \ \mathbf{x}_{1:T})}$$

The latter probabilities allow us to generate new samples starting from $$z_1 \sim p(z_1)$$. In fact, there are more efficient methods for generating such samples (see [Murphy, 2021] for details).

![Path Sampling](/assets/img/hmm/sampling.svg)
<div class="uk-text-center"><b>Path Sampling</b> applied to the coin-tossing problem. Red dots represent true states and blue dots represent sampled states, with y-values jittered for easier visualization. Locations with a higher density of blue dots represent more likely paths condtitional on the observed data.</div>


<!-- ## Learning
Inference assumes that we already know the parameters of the model. For example, is a discrete observation model, it assumes that we know the transition probabilities $$P$$ as well as the observation matrix $$B$$. In some settings, these might be reasonable assumptions. But, in general, we will not know these values, and will instead need to *learn* them from the data.

Learning for hidden Markov models is complicated by the fact that the likelihood of data depends on the hidden state variable. These variables are effectively additional parameters, and their number of scales with the number of observations. Therefore, learning amounts to large-scale optimization.

Fortunately, the structure of the hidden Markov model permits a clever approach to optimization known as expectation-maximization. The key to this approach is to recognize that *if* we know the parameters, then inference provides an exact solution to the optimal latent variables. Furthermore, given beliefs about the latent variables, we can easily maximize likelihood with respect to the parameters. Expectation-maximization works by alternating between these inference and maximization steps, which turns out to force parameter estimates towards their likelihood-maximizing values.

Formally,... -->

<!-- Baum-Welch Algorithm -->
<!-- ```julia

```

### Example: Sneaky Casino


### Example: Boom-Bust Cycle


Hidden Markov models are the simplest for of dynamic probabilistic model that you might think of. There are all sorts of extensions and complications that we might add to the model, for example, adding a "forcing" variable to observations, or permitting autocorrelated observations. You might also think about allowing the hidden variable to have a continuous distribution. We'll look at all of these topics in future posts. -->


### References
[Murphy, 2012] *Machine Learning: a Probabilistic Perspective*, Kevin Murphy.

### Notes