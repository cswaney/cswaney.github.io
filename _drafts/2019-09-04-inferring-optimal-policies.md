---
layout: post
author: Colin Swaney
title: Inferring Optimal Policies
date: 2019-09-04
categories: [research]
category: research
tags: [reinforcement learning]
excerpt: "<p>The standard way to think about reinforcement learning and control is as an optimization problem: define the environment, and I'll come up with an optimal policy. In this post I review an alternative approach to control in which we imagine observing an agent interacting with the environment and infer what their optimal policy must be. The approach is full of subtleties and has fascinating connections to the rest of the world of reinforcement learning. Sounds intriguing?</p>"
---

(**Note**: This post is based on [this lecture](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-15.pdf) by given Sergey Levine in his course on deep reinforcement learning at Berkeley in the fall of 2018).

- Standard optimal control and reinforcement learning using a probabilistic graphical model (PGM) to represent dynamics, but doesn't use associated tools of PGM to find a solution.
- Why would we want to formulate our control problem as a PGM? So that we can use all of the well-known methods for learning and inference!
- "Maximum entropy reinforcement learning" === "exact probabilistic inference (deterministic dynamics)" and "variational inference (stochastic dynamics)"
- Beneifits of this approach?
    1. "Natural" exploration via entropy maximization.
    2. Provides tools for inverse reinforcement learning.
    3. Can use powerful PGM toolkit to solve RL problems.


A motivating story... (learning a task?)


### The Model
- Stochastic dynamcis: $$p(s_{t + 1} \vert s_t, a_t)$$ (unknown)
- Horizon: $$T$$
- Reward: $$r(s_t, a_t)$$
- Policy: $$p(a_t \vert s_t, \theta)$$
- One way to solve this problem is to use a policy gradient algorithm. That will lead to an optimal policy $$p(a_t \vert s_t, \theta^*)$$ and associated optimal trajectory

$$p(\tau) = p(s_1, a_1, \dots, s_T, a_T) \vert \theta^*)$$

- Motivation: "formulate a PGM such that its most probable trajectory corresponds to the trajectory above."

- Optimality variable: $$\mathcal{O}_t = 1$$ if the action at time step $$t$$ is optimal, and zero otherwise.
- We need to relate $$\mathcal{O}_t$$ to $$s_t$$ and $$a_t$$ to complete our  PGM. Choose condtitional likelihood

$$ p(\mathcal{O}_t = 1 \vert s_t, a_t) \propto \exp(r(s_t, a_t))$$

- (This choice is arbitrary, but will simplify equations coming soon).

#### Observation 1
The posterior likelihood of a trajectory conditional on optimality across *all* actions is proportional to the *unconditional* probability of the trajectory times the exponential return.

$$p(\tau \vert \bold{o}_{1:T} = \bold{1}) \propto p(\tau) \exp\left( \sum_{t=1}^T r(s_t, a_t) \right)$$

A corollary is that under deterministic dynamics the most likely trajectory is the same as the optimal trajectory, since $$p(\tau)$$ is the same for all trajectories in that case, and so optimizing the posterior probability above amounts to maximizing its return.


## Policy Search
- The objective in this section is to find the optimal policy, defined as $$p(a_t \vert s_t, \bold{o}_{t:T} = \bold{1})$$, that is, the distribution over actions going forward conditional on each action being optimal.

### Backwards Messages
The upshot is that the policy can be interpreted as proportional to an advantage function in log space.

$$ \log p(a_t \vert s_t, \bold{o}_{t:T}) \propto Q(s_t, a_t) - V(s_t),$$

where

$$ Q(s_t, a_t) = r(s_t, a_t) + \log \mathbb{E}_{s_{t + 1} \sim p(s_{t +1} \vert s_t, a_t)} \left[ \exp(V(s_{t + 1})) \right],$$

and

$$ V(s_t) = \log \int_{\mathcal{A}} \exp(Q(s_t, a_t)) \ d a_t.$$

- $$V$$ and $$Q$$ are "soft" versions of the value and action-value function that we get in standard reinforcement learning.

- In standard reinforcement learning, the value function would be the maximum over the action-value over all actions. Here, it is the so-called "soft maximum", which converges to the maximum as $$Q$$ goes to infinity.

- Conversely, the standard $$Q$$-function would take the expectation over values in the following state. But here we take the soft maximum, which we can think of as an expectation that is weighted towards higher realizations of $$V(s_{t+1})$$. Thus, the policy is an *optimisitc* policy.

## Trajectories

There is something subtle going on that confused me for quite a while before it finally sunk in (it's really not difficult, but does require some clear thinking). Let's look at the distibution of trajectories conditioning on optimality:

$$ p(\tau \vert \bold{o}_{1:T}) = p(s_1 \vert \bold{o}_{1:T}) \prod_{t=1}^{T} p(s_{t+1} \vert s_t, a_t, \bold{o}_{1:T}) p(a_t \vert s_t, \bold{o}_{1:T})$$

If the system is deterministic, then $$p(s_{t+1} \vert s_t, a_t) = p(s_{t+1} \vert s_t, a_t, \bold{o}_{1:T})$$. But if the system is stochastic, then the unconditional transition probabilities are different from their conditional counterparts. Why? If we look at the graphical model, we see that optimality at time $$t$$ doesn't tell us anything about the probability of the next state. But we are conditioning on the *entire* trajectory being optimal. States and optimality are connected in our graph by $$p(\mathcal{O}_t \vert s_t, a_t) = \exp(r(s_t, a_t))$$, which means that knowing that we behave optimally at time $$t$$ tells us something about $$s_t$$ (because $$p(s_t \vert \mathcal{O}_t) \propto p(\mathcal{O}_t \vert s_t) p(s_t)$$). Essentially this boils down to conditional trajectories being weighted toward better states relative to the inherent dynamics.
