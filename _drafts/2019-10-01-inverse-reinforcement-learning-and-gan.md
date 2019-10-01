---
layout: post
author: Colin Swaney
title: Deep Inverse Reinforcement Learning
date: 2019-09-04
categories: [research]
category: research
tags: [reinforcement learning]
excerpt: "<p></p>"
---

**Note**: This post is based on [this lecture](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-16.pdf) given by Sergey Levine in his course on deep reinforcement learning at Berkeley in the fall of 2018.

The typical goal of reinforcement learning is to discover optimal—or at least satisfactory—policies to interact with a given environment. A critical part of the environment is the agent's reward function. It doesn't really matter whether I know the laws of physics if I don't know what task I'm meant to achieve. In constrast, the goal of inverse reinforcement learning is to discover the reward function given a policy function.

Imagine that you've been shown how to perform a task, for example, how to throw a ball. You have many, many examples of an expert performing this task, but now you need to learn the task. For you or I it might be obvious that the task is to hit a target, but your robot doesn't have this intuition. One strategy would be to imitate or mimic the expert behavior. That's a reasonable approach, although there are some [practical difficulties](**TODO**) involved. More importantly, it doesn't actually get to the heart of the matter: we would really like our robot to "understand" her environment so that she can easily learn to perform additional, related tasks.

An alternative approach is for the robot to learn the implicit reward function that the expert policy optimizes. With the its learned reward function, the agent can figure out its own policy and will be able to re-learn optimal policies if its environment is modified in some way. In fact, we can learn the reward function and an associated policy at the same time.

## Maximum Entropy Inverse Reinforcement Learning
Let's take as a starting point the same setting as the [lecture on optimal policies through probabilistic inference](**TODO**). Namely, image that we observe an expert performing a task: we see what states she visits ($$s_t$$), what actions she performs ($$a_t$$), as well as the rewards she receives ($$r_t$$). We imagine a latent variable $$\mathcal{O}_t$$ which codes whether each action performed is optimal: $$\mathcal{O}_t = 1$$ if action $$a_t$$ is optimal in state $$s_t$$, and zero otherwise. We assume that optimal actions are exponentially more likely: $$p(\mathcal{O}_t \vert s_t, a_t) \propto \exp \left( r(s_t, a_t) \right)$$.

The first question is, "What is the probability of an optimal  trajectory?" That is, what is $$p(\tau \vert \mathcal{O}_{1:T})$$? By simply applying Bayes rule we get that

$$p(\tau \vert \mathcal{O}_{1:T}) \propto p(\tau) p(\mathcal{O}_{1:T} \vert \tau) = p(\tau) \prod_{t=1}^T p(\mathcal{O}_t \vert \tau) = p(\tau) \exp \left( \sum_{t=1}^T r(s_t, a_t) \right)$$

![probabilistic-model](**TODO**)

Now return to the inverse reinforcement learning setting. We don't know $$r(s_t, a_t)$$; we want to *learn* $$r_{\psi}(s_t, a_t)$$ that is consistent with the observed trajectories. The basic way to do that is to maximize the likelihood of the trajectory probability above:

$$\max_{\psi} \frac{1}{N} \sum_{i = 1}^{N} \log p(\tau_i \vert \mathcal{O}_{1:T}, \psi) = \max_{\psi} \frac{1}{N} \sum_{i = 1}^{N}  r_{\psi}(\tau_i) - \log Z$$

Where did $$Z$$ come from? Bayes rule tells that $$p(\tau \vert \mathcal{O}_{1:T})$$ is *proportional* to $$p(\tau) \exp \left( \sum_{t=1}^T r(s_t, a_t) \right)$$. To make it a proper probability, we need to normalize the right-hand side so that it integrates to one. $$Z$$ is that normalizer (called the *partition function*), which in this case is given by

$$ Z = \int_{\tau} p(\tau) \exp \left( \sum_{t=1}^T r(s_t, a_t) \right) d \tau $$

This turns out to be the devil in this method. Notice that we are trying to integrate over *all* the trajectories in the environment. In general, this is impossible. But let's consider the case in which the number of states and actions are small and countable.

First, let's take a close look at the gradient of the objective function, $$\mathcal{L} = \frac{1}{N} \sum_{i = 1}^{N} r_{\psi}(\tau_i) - \log Z $$:

$$ \nabla_{\psi} \mathcal{L} = \frac{1}{N} \sum_{i = 1}^{N} \nabla_{\psi} r_{\psi}(\tau_i) - \frac{1}{Z} \nabla_{\psi} Z = \frac{1}{N} \sum_{i = 1}^{N} \nabla_{\psi} r_{\psi}(\tau_i) - \int_{\tau} \left( \frac{p(\tau) \exp \left( r_{\psi}(\tau) \right)}{Z} \right) \nabla_{\psi} r_{\psi}(\tau) d \tau $$

**Comment**: The loss itself is the expectation of the reward under the expert minus the log of $$Z$$. There is actually some intuition here. On the one hand, the expert is assumed to visit state-action pairs that are close to optimal, so we want to encourage our reward function to be high along trajectories the expert visited. On the other hand, we don't want to go assigning probability all over the place. $$Z$$ is sort of the "total probability" associated with $$r_{\psi}$$—we want to minimize this, so its negative in $$\mathcal{L}$$. Also, note that if you know the true reward function, then the gradient of the loss zeros out—the expectations are the same because the policies are the same in that case.

The first term is clearly a sample approximation to the expectation of $$\nabla_{\psi} r_{\psi}$$ under the expert of "demo" policy (sense each of the $$\tau_i$$ are generated by the expert). Looking at the second term, we see that it is *exactly* the expectation of $$\nabla_{\psi} r_{\psi}$$ under the "optimal" policy *according to the reward estimate*, $$r_{\psi}$$. If we know the dynamics of the system, then we can estimate the latter expectation using a forward-backward inference algorithm. (The main idea is that if the states and actions are countable, then we can infer the conditional probability of each state-action pair, then sum up over all state-action pairs to calculate the integral. For details, see [here](TOOD)). Thus, the maximum entropy inverse reinforcement learning algorithms amounts to:

1. Estimate $$ \int_{\tau} \left( \frac{p(\tau) \exp \left( r_{\psi}(\tau) \right)}{Z} \right) \nabla_{\psi} r_{\psi}(\tau) d \tau $$ using the forwards-backwards algorithm.
2. Evaluate $$ \nabla_{\psi} \mathcal{L} $$.
3. $$ \psi \leftarrow \psi + \eta \nabla_{\psi} \mathcal{L} $$.

**TODO**: Why is this called "maximum entropy IRL"? 1. Linear case. 2. The policy from the PGM is a MaxEnt policy...

## Guided Cost Learning
The maximum entropy approach works well for small problems where state-action pairs can be counted and the system dynamics are known (or can be easily learned). In deep reinforcement learning, we are interested in problems with large, continuous state-actions spaces where the dynamics of the environment are unknown. In these settings, we need a different strategy to estimate the part of the loss function that depends on the partition function, $$Z$$. Recall that this term amounted to the expectation of $$\nabla_{\psi} r_{\psi}(\tau)$$ under the distribution of trajectories generated by the optimal policy corresponding to $$r_{\psi}$$. In the probabilistic inference framework, such optimal policies are known as "soft" optimal policies or "maximum entropy policies" because they solve

$$ \max_{\theta} \sum_{t=1}^T \mathbb{E}_{\pi(s_t, a_t \vert \theta)} \left[ r(s_t, a_t) \right] + \mathbb{E}_{\pi(s_t, a_t \vert \theta)} \left[ \mathcal{H}(\theta) \right]$$

This suggests a simple sample estimate: learn an optimal policy under $$r_{\psi}$$ using any MaxEnt RL algorithm you like (see this [post](**TODO**)), sample trajectories $$\{ \tau_j \}_{j=1}^M $$ from that policy, and plug them into the standard estimate:

$$ \mathbb{E}_{} \left[ \nabla_{\psi} r_{\psi}(\tau) \right] \approx \frac{1}{M} \sum_{j=1}^M \nabla_{\psi} r_{\psi}(\tau_j) $$

This results in a loss function that whose first term is an estimate based on samples from the expert and second term is an estimate based on samples from a soft optimal policy. The problem is that now in order to update $$\psi$$ we have to run a full MaxEnt RL algorithm! Clearly this is too expensive—imagine solving your favorite Atari environment a thousand times! What if instead of running MaxEnt RL to completion, we simply run a few steps? The resulting policy won't be the soft optimal one, so we can't directly plug in trajectories to the estimator above. However, we can use importance samplingt to correct for the bias. In particular, the new estimate will be

$$ \frac{1}{\sum_{j} w_j} \sum_{j=1}^M w_j \nabla_{\psi} r_{\psi}(\tau_{j}) $$

where $$w_j = \frac{\exp \left( r_{\psi}(\tau_j) \right)}{\pi(\tau_j)} $$. (Here's a way to understand this estimator. The original thing we wanted to calculate was the expectation of some value assuming a *uniform distribution* over $$\tau$$. Now we weight the trajectories by their probability under $$\pi$$ and simultaneously divide the value by $$\pi$$ so that the overall result is unchanged). The resulting algorithm is called [Guided Cost Learning](**TODO**) because the updates to $$\pi$$ gradually lead to the estimates of $$Z$$ taken from experience that is most relevant.

![guided-cost-learning](**TODO**)

### Algorithm: Guided Cost Learning
Given human/expert demos...
- Initialize random policy, $$\pi$$
Loop...
- Generate samples from $$\pi$$
- Update rewards $$r_{\psi}$$ using samples and demos (using $$\mathcal{L}$$)
- Update $$\pi$$ with respect to $$r_{\psi}$$ using MaxEnt RL algorithm
Output $$r_{\psi}$$ and $$\pi$$.

Notice that the algorithm outputs the reward *and* the (soft) optimal policy associated with that reward. Also, their paper, the authors use a model-based MaxEnt RL algorithm (using LQR) that approximates the dynamics under $$\pi$$. That's important because the algorithm requires us to know the probability of whole trajectories, $$q(\tau)$$, which is one of the outputs of such a model-based approach: we wouldn't get using model-free methods. There are also some regularization tricks involved to make things work nicely... **TODO** In short, this is a highly involved procedure! In the last section we'll see that it is equivalent to a much simpler approach that is more familiar to the general deep learning community.


## Generative Adversarial Imitation Learning
Let's forget inverse reinforcement learning for a minute and return to the idea of imitation learning. We don't care about the reward function; we just want to do what the expert does. We can think of this as learning to perform series of actions that lead to the same distribution of trajectories, $$p(\tau)$$, as the expert. Generative adversarial networks (GANs) [Goodfellow et al.](**TODO**) are a now well-know method for generating samples that mimic complicated distribution. For example, you can train a GAN to generate paintings of a particular style, or you can train a (conditional) GAN to plausibly "fill in" missing information in a photo.

GANs work by playing a game between a "generator" network and a "discriminator" network. The generator tries to trick the discriminator by creating samples that the discriminator can't tell from the real thing, and the discriminator tries to learn the difference between real samples and fakes.



- [Ho and Ermon, NIPS 2016](**TODO**)












## Outline
1. MaxEnt IRL: 
2. Guided Cost Learning
3. GAN
4. Guided Cost Learning = (modified) GAN = MaxEnt Learning

## References
- Ziebart et al., AAAI 2008. *Maximum Entropy Inverse Reinforcement Learning*.
- Finn et al., ICML 2016. *Guided Cost Learning*.
- Ho and Ermon, NIPS 2016. * Generative Adversarial Imitation Learning*.
- Finn et al., 2016. *A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models*.