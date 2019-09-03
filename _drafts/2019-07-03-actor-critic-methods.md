---
layout: post
author: Colin Swaney
title: Actor-Critic Methods
date: 2019-07-03
categories: [research]
category: research
tags: [deep-learning, reinforcement-learning]
sections: [algorithm, implementation, extensions]
---
There are two standard algorithms in reinforcement learning: Q-learning and the policy gradient method. These methods represent orthogonal approaches. Policy gradient methods work on the policy function; Q-learning works on the action-value function. Policy gradient methods use Monte Carlo approximation; Q-learning uses bootstrap updates. And finally, policy gradient methods are low bias, high variance, while Q-learning is a low variance, high bias approach. In statistical learning, there is always some balance of bias and variance that provides the best outcome. Actor-critic methods provide, in some sense, a compromise between Q-learning and policy gradient methods, and can be thought of as a way to balance out the costs and benefits of these approaches.

## Algorithm {#algorithm}
Recall that the vanilla policy gradient algorithm attempts to directly improve a random policy $$\pi$$ by moving in the direction of highest expected return,

$$\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\theta),$$

where $$J(\theta)$$ is the expected return under policy $$\pi_{\theta}$$. The gradient term depends on the policy network probabilities, $$\pi_{\theta}(a_t \vert s_t)$$, and the rewards generated in each episode, $$R_t$$:

$$J(\theta) \approx \sum_{t=1}^T \log \pi_{\theta}(a_t \vert s_t) R_t.$$

The expression above is a very *noisy* estimate of the expected return (because episode returns have a lot of variance), which flows into policy gradient updates. You will therefore hear people say that vanilla policy gradient is a "high variance" method. The standard approach to reduce the variance of the updates is to  subtract a "baseline" from $$R_t$$. A baseline that is itself an estimate of the expected reward can result in a much smaller weights, $$\|R_t - b_t \|   \ll \|R_t\|$$. In other words, subtracting a baseline results in a *re-scaling* of the updates.

Actor-critic methods replace both $$R_t$$ and $$b_t$$ with approximations based on the value function. First, we want $$b_t$$ to be an approximation of the value function, so we replace it by the output of a network  $$\hat{V}^{\pi}_{\phi}(s_t) \approx V^{\pi}(s_t)$$. Second, according to the Bellman equation, the expected reward-to-go is equal to

$$E[R_t] = E \left[r(s_t, a_t) + V^{\pi}(s_t') \right].$$

If we knew the value function, then we could estimate this quantity by evaluating it at random sample transitions $$\{s_t, a_t, r_t, s_t'\}$$. Instead, we add a second approximation error by replacing the true value function with the sample network that we used above. So our estimate of the second term becomes

$$E[R_t] \approx r(s_t, a_t) + \hat{V}_{\phi}^{\pi}(s_t')$$

The overall result is that we've replaced $$R_t - b_t$$ with $$r(s_t, a_t) + \hat{V}_{\phi}^{\pi}(s_t') - \hat{V}_{\phi}^{\pi}(s_t)$$. The latter expression can be seen as an approximation of the *advantage* function, $$A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t),$$ and the resulting algorithm (as described here) is commonly referred to as the "advantage actor-critic" or "A2C" algorithm. We call this a "bootstrap" estimator because it estimates $$V$$ based in part on previous estimate. For comparison, we could also approximate $$E[R_t]$$ using observed rewards,

$$\sum_{\tau = t}^{T} r(s_{\tau}, a_{\tau}),$$

which gives an alternative, Monte-Carlo version of the A2C algorithm.

The A2C algorithm performs two "sub-updates" per full update: one to update the value function, and one to update the policy. The first update is similar to the update performed in the DQN algorithm. There is a slight difference however. For one thing, we're updated the value function instead of the action-value function. More importantly, the network used to find targets is the same network used to make predictions. In the DQN algorithm, we use a "cloned" network to find targets. The second update looks like a policy gradient update, and it is, except that we've approximated the reward and baseline. Therefore it seems reasonable to think of this algorithm as a combination of the two classics. That being said, why don't we also use a "clone" network to perform the value update in the A2C algorithm? (I don't know the answer, so this is really a question I'd like to know the answer to!)

> #### Algorithm (Batch A2C)
> 1. Sample experiences $$\{s_i, a_i, r_i, s'_i\}_{i=1}^N$$ using policy $$\pi_{\theta}(a_i \vert s_i)$$
> 2. Calculate value targets $$\{y_i\}_{i=1}^N$$ defined as $$y_i = r_i + \gamma V_{\phi}^{\pi}(s_i')$$
> 3. Update $$V_{\phi}^{\pi}$$ by minimizing $$\|y_i - V_{\phi}^{\pi}(s_i)\|^2_2$$
> 4. Calculate policy targets $$\{A_i\}_{i=1}^N$$ defined as $$A_i = y_i - V_{\phi}^{\pi}(s_i   )$$
> 5. Update $$\pi_{\theta}(a_i \vert s_i)$$ by maximizing $$\sum_{i=1}^N \log \pi_{\theta}(a_i \vert s_i) A_i$$

**Notes** 1. In step (3) the targets $$y_i$$ are treated as constants (even though they depend on the value network). 2. In step (4) we are using the updated value network to re-calculate $$y_i$$ and $$V_{\phi}^{\pi}(s_i)$$. 3. In step (5) the targets $$A_i$$ are treated as constants (or else $$\phi$$ will also update--we only want to update $$\theta$$ in this step).

### Aside: Variance and Bias
I'd like to point out a few facts that help to compare the actor-critic method outlined above with the policy gradient method it mirrors. First, I want to say that the policy gradient algorithm is *unbiased*, while the actor-critic method is (in general) *biased*. What I mean by this statement is this: both algorithms work by approximating the expected return of policy $$\pi$$, and using that approximation to decide how to improve the policy. The value that we use to approximate the value we really want is called "unbiased" if it equals the true value on average; otherwise, its biased. The practical way to think about this is that unbiased methods will eventually bring you arbitrarily close to the truth if you collect enough data: if I played enough games, I would eventually get a very good idea of the expected return of $$\pi$$. A biased method method will *always* leave some room for improve--even at infinity.

Now for the second claim: the policy gradient approach has a higher variance than the actor-critic method. What do I mean by that? Essentially that the value I am using to approximate the true value I'm interested is expected to vary more under the policy gradient approach than under the actor-critic approach. The reason is that the policy gradient approach uses the full reward-to-go, which potentially varies *a lot*, depending on the environment, whereas the actor-critic method uses a value that is typically close to zero (because they are regression residuals). The rest of the approximation is identical.

Why does all this matter? *All* statistical learning--to the extent that it attempts to learn the value of some function--is subject to a "bias-variance tradeoff", whereby methods which lower bias tend to have higher variance, and vice-a-versa. What method works best in any particular application depends on what data is available, and, generally speaking, when there isn't that much data available, low variance methods work better. In reinforcement learning, we're really in a low-data environment when we consider the complexity of the system the agent is trying to learn and the limited number of times that it can interact with that environment. As an analogy, if our task was instead to estimate a regression model in a standard supervised learning task, if we are given only a small amount of data, then it may well be that a simple linear model outperforms nonlinear alternatives because those methods overfit small samples. The same thing is principle is at work here: we are accepting bias in our estimate in exchange of reduced variance, and the result tends to perform better, empirically (on interesting problems).

## Implementation {#implementation}
- In this implementation I'll use two networks, one to represent the actor ($$\pi$$), and one to represent the critic ($$V$$). It's also possible to train a single network with two branches. This might make sense if the network contains convolutional layers so as to provide each branch with the same visual information.
- The actor-critic algorithm is similar to the basic policy gradient algorithm, so we can mostly use the same routines from before. The main "gotcha" is making sure that training operations modify the correct parameters based on the correct errors. In step (2), we calculate targets $$y_i$$ using the value network $$V_{\phi}^{\pi}$$. In the following step we want to update the parameters $$\phi$$ based on $$V_{\phi}^{\pi}(s_i')$$, treating the $$y_i$$ as fixed. The simple way to do this is to calculate the targets and the update in separate steps:
```python
targets = sess.run(values, feed_dict={states_pl: states})
sess.run(values_update, feed_dict={states_pl: next_states, targets_pl: targets})
```
- Now once again in the policy update, $$A_i$$ involves the value network, so an update that directly used the $$A_i$$ *tensor* would update $$\theta$$ and $$\phi$$. We can do the same thing as before, pre-calculating the weights $$A_i$$, and then feeding these as constants in the policy update operation. First, we get new $$y_i$$ targets based on the updated value network
```python
targets = sess.run(values, feed_dict={states_pl: states}),
```
then we calculate the weights and update the policy network
```python
weights = sess.run(weights, feed_dict={states_pl: next_states})
sess.run(policy_update, feed_dict={states_pl: states, actions_pl: actions, weights_pl: weights})
```
- The rest of the implementation basically follows the policy gradient implementation.  


## Extensions {#extensions}

### Step Size
I mentioned an alternative Monte Carlo based approach earlier that simply uses the observed reward-to-go values, and I also mentioned that the actor-critic method is in some sense a combination of Q-learning and policy gradient with baseline. Now the question is, can I somehow vary the degree of the combination? Instead of approximating the reward-to-go by $$r(s_t, a_t) + \gamma V_{\phi}^{\theta}(s_{t + 1})$$, I could expand the approximation one step and use

$$r(s_t, a_t) + \gamma r(s_{t + 1}, a_{t + 1}) + \gamma^2 V_{\phi}^{\pi}(s_{t + 2})$$

If I kept replacing $$V$$ terms like this until I reached the end of the episode, then I'd end up with the full reward-to-go! In other words, as I increase the expansion my update gradually turns into the Monte Carlo algorithm. In terms of our implementation, the only difference is that we need to keep track of $$N$$-step transitions, and feed these to the graph--the graph itself is exactly the same. Small values of $$N$$ (e.g., $$N = 3, 4$$) can help stabilize the A2C learning curve in the CartPole environment the same way they help stabilize DQN.

### Memory Replay
DQN uses another trick to stabilize the training process, which is to keep a memory of transitions instead of just using the most recent `batch_size` transitions to update the value function. Can we use the same trick to stabilize A2C?

### Entropy
[Mnih et al. (2016)]() include the entropy of the policy as an additional term in their policy update. The idea is to *encourage* policies to be more random than they would be otherwise, which can be seen as a way to encourage exploration (entropy is maximized by a uniform random variable--that is, a policy that picks actions at random with equal probability). The update now maximizes the quantity

$$\sum_{i=1}^N \log \pi_{\theta}(a_i \vert s_i) A_i + \beta H(\pi_{\theta}(s_t)),$$

where $$H(\pi_{\theta}(s_t))$$ is the entropy of the policy,

$$H(\pi_{\theta}(s_t)) = - \sum_{k=1}^K \pi_{\theta}(a_t = k \vert s_t) \log \pi_{\theta}(a_t = k \vert s_t)$$

In Tensorflow, we adjust our policy update as follows:
```python
entropy_loss = beta * tf.reduce_mean(
    tf.multiply(
        tf.nn.softmax(policy_logits),  # probabilities
        tf.nn.log_softmax(policy_logits)  # log probabilities
    )
)
policy_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss + entropy_loss)
```

Note that there is no negative sign in front of the entropy calculation because we want to *maximize* entropy, and therefore we need to minimize *negative* entropy.
