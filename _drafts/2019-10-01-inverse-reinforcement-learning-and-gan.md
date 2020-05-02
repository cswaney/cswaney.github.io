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

> **Note**: This post is based on [a lecture](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-16.pdf) given by Sergey Levine in his course on deep reinforcement learning at Berkeley in the fall of 2018.

The typical goal of reinforcement learning is to discover optimal—or at least satisfactory—policies to interact with a given environment. A critical part of the problem is the agent’s reward function, which motivates the agent to perform optimal actions. We can’t merely provide intelligent agents with the laws of physics and expect them to perform interesting and desirable behaviors: we need to be able to communicate the tasks they are meant to achieve. Designing such reward functions by hand is extremely challenging, and can lead to entirely unexpected outcomes. The idea of inverse reinforcement learning is to learn reward functions by observing experts that already know how to perform a given task.

Imagine that we want to teach a robot a new task, for example, how to throw a ball so that it hits a target. We have collected many samples of experts performing this task, and we want to transfer that knowledge to the robot. For you or I, it is obvious that the task is to hit the target, but the robot doesn’t have this intuition. One strategy is to imitate or mimic the expert’s behavior. That’s a reasonable approach, but it doesn’t get to the heart of the matter. We want our robot to “understand” its environment so that it can quickly learn to perform additional, related tasks.

An alternative approach is to infer the implicit reward function that the expert policy optimizes and train our robot using standard reinforcement learning algorithms. With a learned reward function, the agent can figure out a  policy and will be able to re-learn optimal policies if we modify its environment. We can even learn the reward function and an associated policy at the same time.

## Maximum Entropy Inverse Reinforcement Learning
Let's take as a starting point the same setting as the [this post]({% post_url 2019-09-04-inferring-optimal-policies %}) on learning optimal policies through probabilistic inference. In this setting, we observe an expert performing a task numerous times. We see what states she visits ($$s_t$$), what actions she performs ($$a_t$$), as well as the rewards she receives, $$r(s_t, a_t)$$. We imagine a latent variable $$\mathcal{O}_t$$ which codes whether each action performed is optimal: $$\mathcal{O}_t = 1$$ if action $$a_t$$ is optimal in state $$s_t$$, and zero otherwise. We assume that optimal actions are exponentially more likely such that

$$p(\mathcal{O}_t \vert s_t, a_t) \propto \exp \left( r(s_t, a_t) \right)$$

The figure below shows the graphical model for this scenario.

![graphical-model](/assets/img/graphical-model.png)

Let's return to the inverse reinforcement learning setting. Here, we don't know $$r(s_t, a_t)$$; we want to learn a parameterized estimate $$r_{\psi}(s_t, a_t)$$ that is consistent with our observations. The basic way to do that is to maximize the likelihood of the trajectories assuming that the agent acts optimally at each step:

$$\max_{\psi} \frac{1}{N} \sum_{i = 1}^{N} \log p(\tau_i \vert \mathcal{O}_{1:T}, \psi)$$

We are immediately presented with the question, "What is the probability of an optimal trajectory?" By applying Bayes rule, we find that

$$p(\tau \vert \mathcal{O}_{1:T}) = \frac{p(\tau)}{Z} p(\mathcal{O}_{1:T} \vert \tau) = \frac{p(\tau)}{Z} \prod_{t=1}^T p(\mathcal{O}_t \vert \tau) = \frac{p(\tau)}{Z} \exp \left( \sum_{t=1}^T r_\psi(s_t, a_t) \right)$$

Plugging into the maximum likelihood objective above (and removing terms that don't depend on $$\psi$$) we get

$$\max_{\psi} \frac{1}{N} \sum_{i = 1}^{N}  r_{\psi}(\tau_i) - \log Z$$

Where did $$Z$$ come from (and why does it depend of $$\psi$$)? Bayes rule tells that $$p(\tau \vert \mathcal{O}_{1:T})$$ is *proportional* to $$p(\tau) p(\mathcal{O}_{1:T} \vert \tau)$$. To make this a proper probability, we need to normalize the right-hand side so that it integrates to one. The *partition function*, $$Z$$, is the normalizer, which in this case is given by

$$ Z = \int_{\tau} p(\tau) \exp \left( \sum_{t=1}^T r_\psi(s_t, a_t) \right) d \tau $$

This isn't a pretty integral. Notice that we are trying to integrate over *all* the trajectories in the environment. In general, this is impossible, so let's consider the case in which the number of states and actions are small and countable.

Let's first take a closer look at the gradient of the objective function, 

$$\nabla_\psi \mathcal{L} = \nabla_\psi \left( \frac{1}{N} \sum_{i = 1}^{N} r_{\psi}(\tau_i) - \log Z \right) = \nabla_{\psi} \frac{1}{N} \sum_{i = 1}^{N} r_{\psi}(\tau_i) - \frac{1}{Z} \nabla_{\psi} Z$$

Pushing the gradients inside the sum and integral we get

$$\nabla_{\psi} \mathcal{L} = \frac{1}{N} \sum_{i = 1}^{N} \nabla_{\psi} r_{\psi}(\tau_i) - \frac{1}{Z} \int_{\tau} p(\tau) \exp \left( r_{\psi}(\tau) \right) \nabla_{\psi} r_{\psi}(\tau) \ d \tau$$

The first term is clearly a sample approximation to the expectation of $$\nabla_{\psi} r_{\psi}$$ under the expert policy:

$$\frac{1}{N} \sum_{i = 1}^{N} \nabla_{\psi} r_{\psi}(\tau_i) \approx \mathbb{E}_{\pi^*} \left[ \nabla_{\psi} r_{\psi}(\tau_i) \right]$$

Looking at the second term, we see that it is *exactly* the expectation of $$\nabla_{\psi} r_{\psi}$$ under the "optimal" policy given the reward estimate, $$\pi_\psi^*$$:

$$\frac{1}{Z} \int_{\tau} p(\tau) \exp \left( r_{\psi}(\tau) \right) \nabla_{\psi} r_{\psi}(\tau) d \tau = \int_{\tau} p(\tau \vert \mathcal{O}_{1:T}) \nabla_{\psi} r_{\psi}(\tau) d \tau = \mathbb{E}_{\pi_\psi^*} \left[ \nabla_\psi r_\psi (\tau) \right]$$

If we know the dynamics of the system and the combined state-action space is small and countable, then we can estimate this expectation by first estimating the probability of each state-action pair, $$p(s_t, a_t \vert \mathcal{O}_t)$$. In a [previous post]({% post_url 2019-09-04-inferring-optimal-policies %}), we saw that 

$$p(s_t, a_t \vert \mathcal{O}_{1:T}) \propto \beta(s_t, a_t) \alpha(s_t),$$

where $$\alpha_t$$ and $$\beta_t$$ are the forward and backward messages from a standard forward-backward inference algorithm (e.g., for performing inference in a hidden Markov model). Thus, we can estimate the integral by

$$\mathbb{E}_{\pi_\psi^*} \left[ \nabla_\psi r_\psi (\tau) \right] \approx \sum_{t = 1}^T \sum_{(s_t, \ a_t)} \mu(s_t, a_t) \nabla_\psi r(s_t, a_t),$$

where $$\mu(s_t, a_t) \propto \beta(s_t, a_t) \alpha(s_t)$$ estimates the conditional probability of each state-action pair (normalized by summing over all state-action pairs). For details, see [Ziebart et al., 2008](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf).

> **Comment**: The loss itself is the expectation of the reward under the expert minus the log of $$Z$$. There is some intuition here. On the one hand, the expert is assumed to visit state-action pairs that are close to optimal, so we want to encourage our reward function to be high along trajectories the expert visited. On the other hand, we don't want to go assigning probability all over the place. $$Z$$ is the "total probability" associated with $$r_{\psi}$$—we want to minimize this, so it is negative in $$\mathcal{L}$$. Also, note that if you know the true reward function, then the gradient of the loss zeros out—the expectations are the same because the policies are the same in that case.

Let's summarize. We assume that the expert performs optimal actions with respect to some reward function that we parameterize cleverly. We learn the parameters of the reward function using maximum-likelihood estimation, where we use a forward-backward algorithm to estimate the partition function, and estimate the remaining part using the expert demonstrations. This procedure amounts to the following algorithm.

> #### Algorithm: Maximum Entropy Inverse Reinforcement Learning
> 1. Generate expert demonstrations, $$\tau_i$$.
> 2. Initialize a random reward function, $$r_\psi$$.
> 3. Repeat:
>   - Estimate $$\nabla_\psi \log Z$$ using forward-backward inference.
>   - Perform gradient ascent on $$\psi$$ according to $$\nabla_\psi \mathcal{L}$$.

<!-- code -->

Why is this "maximum entropy" inverse reinforcement learning? One interpretation is that the algorithm is the result of maximizing the entropy of the distribution of trajectories subject to a "feature-matching" constraint. That is, we find a reward whose optimal policy generates the most random trajectories possible while visiting states with the same frequency as the expert:

$$\max_\psi \mathcal{H(\pi_\psi)} \ s.t. \ \mathbb{E}_{\pi_\psi}\left[ \mathbf{f} \right] = \mathbb{E}_{\pi^*} \left[ \mathbf{f} \right],$$

where we assume that $$r_\psi(s, a) = \theta^\top \mathbf{f}$$. Another way to think about this is that the policy used to estimate the partition function corresponds to a soft optimal—or "maximum entropy"—policy, because we derived it from the probabilistic inference framework.


## Guided Cost Learning
The maximum entropy approach works well for small problems where state-action pairs can be counted and the system dynamics are known (or can be easily learned). In deep reinforcement learning, we are interested in problems with large, continuous state-actions spaces where the dynamics of the environment are unknown. In these settings, we need a different strategy to estimate the part of the loss function that depends on the partition function, $$Z$$. Recall that this term amounted to the expectation of $$\nabla_{\psi} r_{\psi}(\tau)$$ under the distribution of trajectories generated by the optimal policy corresponding to $$r_{\psi}$$. In the probabilistic inference framework, such optimal policies are known as "soft" optimal policies or "maximum entropy policies" because they solve

$$ \max_{\theta} \sum_{t=1}^T \mathbb{E}_{\pi(s_t, a_t \vert \theta)} \left[ r(s_t, a_t) \right] + \mathbb{E}_{\pi(s_t, a_t \vert \theta)} \left[ \mathcal{H}(\theta) \right]$$

This suggests a simple sample estimate: learn an optimal policy under $$r_{\psi}$$ using any MaxEnt RL algorithm you like (see this [post](**TODO**)), sample trajectories $$\{ \tau_j \}_{j=1}^M $$ from that policy, and plug them into the standard estimate:

$$ \mathbb{E}_{\tau_j \sim q_\psi} \left[ \nabla_{\psi} r_{\psi}(\tau) \right] \approx \frac{1}{M} \sum_{j=1}^M \nabla_{\psi} r_{\psi}(\tau_j) $$

This results in a loss function that whose first term is an estimate based on samples from the expert and second term is an estimate based on samples from a soft optimal policy. The problem is that now in order to update $$\psi$$ we have to run a full MaxEnt RL algorithm! Clearly this is too expensive—imagine solving your favorite Atari environment a thousand times! What if instead of running MaxEnt RL to completion, we simply run a few steps? The resulting policy won't be the soft optimal one, so we can't directly plug in trajectories to the estimator above. However, we can use importance sampling to correct for the bias.

This works as follows. Recall that the integral that we want to estimate is

$$Z = \int \exp \left( r_\psi(\tau) \right) d \tau$$

In importance sampling, we convert this integral into an expectation under a different distribution ($$Z$$ is essentially an expectation under a uniform distribution over $$\tau$$):

$$Z = \int \exp \left( r_\psi(\tau) \right) d \tau = \int \frac{\exp \left( r_\psi(\tau) \right)}{q(\tau)} q(\tau) d \tau =  \mathbb{E}_{\tau \sim q} \left[ \frac{\exp \left( r_\psi(\tau) \right)}{q(\tau)} \right]$$

This is just algebra—it's true for *any* $$q$$—but choosing $$q(\tau) \propto \vert \exp \left( r_\psi(\tau) \right) \vert$$ turns out to be optimal. The expectation under $$q$$ is now easy to estimate:

$$ Z \approx \frac{1}{M} \sum_j \frac{\exp \left( r_\psi(\tau) \right)}{q(\tau)} $$

Moreover, we can calculate this approximation's gradient:

$$\nabla_\psi Z \approx = - \frac{1}{M} \sum_j \frac{\exp \left( r_\psi(\tau) \right)}{q(\tau)} \nabla_\psi r_\psi(t_j)$$

Thus our new importance sampling estimate is

$$ \nabla_\psi \log Z = \frac{1}{Z} \nabla_\psi Z \approx \frac{1}{\sum_{j} w_j} \sum_{j=1}^M w_j \nabla_{\psi} r_{\psi}(\tau_{j}) $$

where $$w_j = \frac{\exp \left( r_{\psi}(\tau_j) \right)}{\pi(\tau_j)} $$. This formula provides an efficient estimate of the second term of the loss function even though we have only taken a few steps (or perhaps just a *single* step) in the direction of the "correct" policy. This results in an algorithm which iterative improves estimates of the partition function $$Z$$, reward $$r_\psi$$, *and* optimal policy $$\pi$$ known as  [Guided Cost Learning](https://arxiv.org/abs/1603.00448).

> ### Algorithm: Guided Cost Learning
> 1. Generate expert demonstrations, $$\tau_i$$.
> 2. Initialize a random policy network $$\pi_\theta$$ and reward network $$r_\psi$$.
> 3. Repeat:
>   - Run policy $$\pi_\theta$$ to collect trajectories $$\tau_j$$.
>   - Perform gradient ascent on $$\psi$$ according to $$\nabla_\psi \mathcal{L}$$.
>   - Update $$\pi_\theta$$ by running a policy optimization algorithm.

Notice that the algorithm outputs the reward as well as the soft optimal policy associated with that reward. The paper contains a number of details that make things work nicely. For example, the authors use a model-based maximum entropy reinforcement learning algorithm that uses LQR to approximate the dynamics under the optimal policy. That seems important because the algorithm requires us to know the probability of whole trajectories, $$q(\tau)$$. We wouldn't get using model-free methods. They also use some interesting regularization methods to improve their results. Overall, it is a much more complex algorithm than what we might be used to in deep reinforcement learning! Next, we'll see that guided cost learning is equivalent to a simpler approach that should be more familiar to the general deep learning community.

![guided-cost-learning](/assets/img/guided-cost-learning.png)

> **Comment**: This is similar to an idea from a classical optimal control, namely, value iteration. In value iteration, we alternate between updating an estimate of the value function and an optimal policy with respect to that value function.

## Generative Adversarial Imitation Learning
Let's forget inverse reinforcement learning for a minute and return to the idea of imitation learning. We don't care about the reward function; we just want to do what the expert does. We can think of this as learning to perform series of actions that lead to the same distribution of trajectories, $$p(\tau)$$, as the expert. Generative adversarial networks (GANs) [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661) are a now well-know method for generating samples that mimic complicated distribution. For example, you can train a GAN to generate paintings of a particular style, or you can train a (conditional) GAN to plausibly "fill in" missing information in a photo.

![gan](/assets/img/gan.png)

GANs work by playing a game between a "generator" network and a "discriminator" network. The generator tries to trick the discriminator by creating samples that the discriminator can't tell from the real thing, and the discriminator tries to learn the difference between real samples and fakes. More specifically, let $$G_{\theta}(z)$$ be the generator network that maps noise $$z$$ to the output space, and let $$D_{\psi}(x)$$ be the discriminator that maps the output space to the probability that an example output is real. We train the networks by creating samples that are equal parts real and fake outputs, update $$D$$ by maximizing the probability of labeling the outputs correctly, and update $$G$$ by maximizing the probability of labeling the fake outputs *incorrectly*.

> #### Algorithm: Generative Adversarial Learning
> 1. Generate samples from the truth distribution, $$p(x)$$.
> 2. Randomly initialize networks $$D_\psi(x) $$ and $$G_\theta(x)$$.
> 3. Repeat:
>     - Generate $$N$$ samples using $$G$$ and randomly choose $$N$$ of the true samples.
>     - Perform a gradient ascent update on $$D_\psi$$ according to $$\nabla_{\psi} \mathcal{L}_D$$.
>     - Perform a policy update on $$G_\theta$$ according to $$\nabla_{\theta}\mathcal{L}_G$$.

The loss of the discriminator update is

$$\mathcal{L}_{D} \approx \frac{1}{N} \sum_{i=1}^N \log D_{\psi}(x_i) + \sum_{j=1}^N \log (1 - D_{\psi}(G_{\theta}(z_j))),$$

the loss of the generator is

$$\mathcal{L}_{G} \approx \frac{1}{N} \sum_{i=1}^N \log D(G_{\theta}(z_i)) - \sum_{i=1}^N \log (1 - D(G_{\theta}(z_i))),$$

and the updates are:

$$ \psi \leftarrow \psi + \eta \nabla_{\psi} \mathcal{L}_D $$

$$ \theta \leftarrow \theta + \eta \nabla_{\theta} \mathcal{L}_G $$


[Ho and Ermon, NIPS 2016](https://arxiv.org/abs/1606.03476) showed that GANs provide an effective framework for performing imitation learning. The idea is that the discriminator distinguishes between expert and sample trajectories, while the generator tries to "improve" its policy by making it less likely to be "found out". Typically the policy is trained using a version of deep policy gradient, but the normal reward is replaced by the probability of generating samples that look like the expert, i.e.

$$ \nabla_\theta \mathcal{L}_\pi \approx \frac{1}{M} \nabla_\theta \log \pi_\theta (\tau_j) \log D_\psi (\tau_j) $$

> #### Algorithm: Generative Adversarial Imitation Learning
> 1. Generate samples from an expert policy, $$\pi^{*}$$.
> 2. Randomly initialize networks $$D_{\psi}(\tau) $$ and $$ \pi_{\theta}(a \vert s)$$.
> 3. Repeat:
>     - Generate $$N$$ samples from $$\pi_{\theta}$$ and choose $$N$$ of the expert samples.
>     - Perform a gradient ascent update on $$D_\psi$$ according to $$\nabla_{\psi} \mathcal{L}_D$$.
>     - Perform a policy update on $$\pi_\theta$$ according to $$\nabla_{\theta}\mathcal{L}_\pi$$.

This is all well and good, but it isn't inverse reinforcement learning: in the end, we recover a policy that mimics the expert, but we don't really know how we got there. With a slight modification, however, we can leverage the GAN framework to learn a policy and a reward function. This works as follows.

Suppose that we know the expert distribution $$l(\tau)$$ and the imitation distribution $$q(\tau)$$. What probability should we assign to $$\tau$$ being generated by the expert? The answer is the conditional probability,

$$D(\tau) = \frac{l(\tau)}{l(\tau) + q(\tau)}$$

All we need to do is parameterize $$l(\tau)$$ using the maximum entropy approach:

$$l(\tau) = p(\tau) \frac{1}{Z} \exp \left( r_{\psi}(\tau) \right)$$

We can also note that the likelihood of the generator is always given by

$$q(\tau) = p(\tau) \prod_t \pi_{\theta}(a_t \vert s_t)$$

Thus, after canceling out $$p(\tau)$$, we find that the optimal discriminator is 

$$D(\tau) = \frac{ \frac{1}{Z} \exp \left( r_{\psi}(\tau) \right)}{ \frac{1}{Z} \exp \left( r_{\psi}(\tau) \right) +  \prod_t \pi_{\theta}(a_t \vert s_t)} $$

The generative adversarial approach to *inverse* reinforcement learning uses this discriminator and updates the generator policy using $$r_{\psi}$$ instead of $$D$$:

$$ \nabla_{\theta} \mathcal{L}_{G} \approx \frac{1}{M} \sum_{j=1}^M \nabla_{\theta} \log \pi_{\theta} (\tau_j) r_{\psi}(\tau_j) $$

The discriminator's loss is the same as before:

$$ \mathcal{L}_{D} \approx \frac{1}{N} \sum_{i=1}^N \log D(\tau_i) + \frac{1}{N} \sum_{j=1}^N \log (1 - D(\tau_j)), $$

where $$\tau_i$$ are trajectories generated by the expert and $$\tau_j$$ are trajectories generated by the agent.

> #### Algorithm: Generative Adversarial Inverse Reinforcement Learning
> 1. Generate samples from an expert policy, $$\pi^{*}$$.
> 2. Randomly initialize networks $$D_{\psi, Z}(\tau) $$ and $$ \pi_{\theta}(a \vert s)$$.
> 3. Repeat:
>     - Generate $$N$$ samples from $$\pi_{\theta}$$ and choose $$N$$ of the expert samples.
>     - Perform a gradient ascent update on $$D_{\psi, Z}$$ according to $$\nabla_{\psi, Z} \mathcal{L}_D$$.
>     - Perform a policy update on $$\pi_\theta$$ according to $$\nabla_{\theta}\mathcal{L}_\pi$$.


Did you notice any similarity between the figures representing the guided cost learning algorithms and GAN? [Finn et al. 2016](https://arxiv.org/abs/1611.03852) shows that this algorithm is infact *equivalent* to maximum entropy inverse reinforcement learning, and guided cost learning in particular. Notice that the discriminator network has two trainable parameters, $$\psi$$ and $$Z$$. $$\psi$$ parameterizes the reward function, $$r_\psi$$, and $$Z$$ is a *direct* estimate of the partition function. In this setting we don't have to mess with importance sampling because we directly learn $$Z$$! Also notice that there are actually two changes required to equate GAN with maximum entropy IRL. First, we assume that we have access to $$q(\tau)$$ so that we can evaluate $$D_{\psi, Z}(\tau)$$. Second, we parameterize $$l$$ in terms of the reward function, $$r_\psi(\tau)$$. These changes allow us to learn the reward function instead of only learning a binary classification probability.

> **Question**: Does it matter whether we use the reward or the discriminator to update $$\pi_\theta$$? 

I hope this post has done some justice to this fascinating area of study. The big picture of inverse reinforcement learning is that we want to learn the underlying motivation of agents that we assume behave near optimally. It strikes me that these methods might have applications outside of the robotics domain. For example, researchers could use such methods to develop task-automation tools that mimic user preferences. You can learn more by reading the papers below.

## References
- Ziebart et al., AAAI 2008. *Maximum Entropy Inverse Reinforcement Learning*.
- Finn et al., ICML 2016. *Guided Cost Learning*.
- Goodfellow et al., NIPS 2014. *Generative Adversarial Networks*.
- Ho and Ermon, NIPS 2016. *Generative Adversarial Imitation Learning*.
- Finn et al., 2016. *A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models*.