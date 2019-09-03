---
layout: post
author: Colin Swaney
title: Classical Reinforcement Learning
date: 2019-06-08
categories: [research]
category: research
tags: [reinforcement learning]
---

"Old school" approaches to reinforcement learning (dynamic programming).

To the uninitiated, deep reinforcement learning is sort of magical. Through some sorcery, we tell the computer to play a video game, and a few hours (or days) later its beating the best humans in the world. Before diving into these algorithms, it's worth taking a moment to review some of the core ideas and methods that provide the foundation for modern reinforcement learning. There are basically two central methods in reinforcement learning: dynamic programming, and Monte Carlo methods. Most of the recent advances in reinforcement learning involve some combination of these two methods combined with neural networks and a bunch of clever tricks to try to stabilize training and speed up convergence.

## Dynamic Programming
Let's start with dynamic programming. Remember that our goal is to come up with an optimal policy, and that an optimal policy is one that generates the highest return on average, a quantity we're calling the *value* of the policy. In the last post, I introduced the Bellman equation

$$ V^{\pi}(s) = \sum_a \pi(a \ \vert \ s) \sum_{r_t, \ s_{t + 1}} p(r_t, s_{t + 1} \ \vert \ s_t = s, a_t = a) \left[ r_t + \gamma V^{\pi}(s_{t + 1}) \right] $$

The right way to think about this equation is that it tells us the $$V^{\pi}$$ is a *fixed-point*. Fixed-points are nice because we can typically find them through a fixed-point iteration algorithm. If $$x^{\ast}$$ is a fixed-point of $$f$$ (so that $$f(x^{\ast}) = x^{\ast}$$), then $$x_{k + 1} \leftarrow f(x_k)$$ converges to $$x^{\ast}$$. This means that for any policy $$\pi$$ we might be interested in, we can determine its value to arbitrary precision by turning the Bellman equation into a fixed-point iteration. Think of $$V^{\pi}$$ as a vector with one element per state or world. Say we start out by setting all elements of $$V^{\pi}$$ to zero and call this vector $$V_0$$. Now we'll write out the Bellman equation as an element-wise update rule for this vector:

$$ V_k[s] = \sum_a \pi(a \ \vert \ s) \sum_{r, \ s'} p(r, s' \ \vert \ s_t = s, a_t = a) \left[ r + \gamma V_{k + 1}[s'] \right] $$

Under pretty mild conditions these iterations will converge to the value of our policy, providing a simple tool to determine how good any policy is.

```julia
# Algorithm (Policy Evaluation)
function policy_evaluation(V0, pA, pR, pS, R, gamma; tol::Float64=1e-6)
    """Evaluate policy via dynamic programming.

        - `V0`: initial guess of value.
        - `pA`: array of action probabilities, p(a | s).
        - `pR`: array of reward probabilities, p(r | s, a).
        - `pS`: array of action probabilities, p(s' | s, a).
        - `R`: array of reward values *corresponding* to p(r | s, a).

        # Returns
        - `V`: vector of policy values, V(s).
    """

    V = copy(V0)
    steps = 0
    delta = 0.0
    while true
        delta = 0.0
        for (v_idx, v) in enumerate(V)
            v_update = 0.0
            for (a_idx, pa) in enumerate(pA[v_idx, :])
                for (r_idx, pr) in enumerate(pR[v_idx, a_idx, :])
                    r = R[r_idx]
                    for (s_idx, ps) in enumerate(pS[v_idx, a_idx, :])
                        v_update += pa * pr * ps * (r + gamma * V[s_idx])
                    end
                end
            end
            V[v_idx] = v_update
            # println("V[$v_idx] = $(V[v_idx])")
            delta = max(delta, abs(v - v_update))  # update maximum change
        end
        # println(V)
        steps += 1
        # println("delta=$delta")
        if delta < tol
            break
        end
    end
    # println("converged in $steps steps (delta=$delta)")
    return V
end
```

Finding the value of a policy is all good and well, but we want to improve our policy. How do we do that? Now that we know the value of the policy in any possible future state of the world, we can calculate the expected return from performing any available action, and then following our current policy after that, which is exactly what I defined as the action-value in a previous post:

$$ Q^{\pi}(s, a) = \mathbb{E}_{\pi} \left[ r_t + \gamma V^{\pi}(s_{t + 1}) \ \vert \ s_t = s, a_t = a \right] $$

Suppose for a second that we only consider deterministic policies. Now if some action produces a higher action-value than the value of our policy in *some* state $$s$$, then we can improve our policy by choosing that action in that state instead of the action specified by our policy. The reason is clear enough: in every other state of the world we still get the same value, and we get a higher value in state $$s$$. To improve our policy further, we can choose the action that maximizes the action-value, and we can repeat this update for each state:

```julia
# Algorithm (Policy Improvement)
function policy_improvement(V, nactions, pR, pS, R, gamma)
    policy = zeros(length(V), nactions)  # policy[s, a] = π(a | s)
    for (v_idx, v) in enumerate(V)
        a_max = 0
        Q_max = -Inf
        for a_idx in 1:nactions
            Q = 0.0
            for (s_idx, ps) in enumerate(pS[v_idx, a_idx, :])
                v_next = V[s_idx]
                for (r_idx, pr) in enumerate(pR[v_idx, a_idx, :])
                    r = R[r_idx]
                    Q += ps * pr * (r + v_next)
                end
            end
            if Q > Q_max
                a_max = a_idx
                Q_max = Q
            end
        end
        # println("state={$v_idx}, new action={$a_max}")
        policy[v_idx, a_max] = 1.0
    end
    return policy
end
```

Putting the policy evaluation and policy improvement steps together gives a solution to our problem. All we need to do is iterate between learning the value of a policy and improving that policy:

```julia
# Algorithm (Policy Iteration)
function policy_iteration(policy, pR, pS, R, gamma, tol=1e-6)
    nstates, nactions = size(policy)
    value = zeros(nstates)
    i = 0
    while true
        value_new = policy_evaluation(value, policy, pR, pS, R, gamma, tol=tol)
        policy = policy_improvement(value_new, nactions, pR, pS, R, gamma)
        i += 1
        println("iter=$i, max_diff=$(maximum(abs.(value_new - value)))")
        if isapprox(value, value_new)
            break
        else
            value = value_new
        end
    end
    return policy, value
end
```

(Note that we don't need to start out our policy evaluation procedure from scratch each iteration. The value of our new policy is likely to be closer to the value of our previous policy than it is to zero, so we can initialize our value to the value found in the previous step). It turns out that we don't need to actually run the policy evaluation step to convergence each iteration. In fact, it will be good enough to run a *single* step of the policy evaluation procedure! In addition, the policy improvement algorithm always results in a greedy, *deterministic* policy. As a result, computing the expectation $$\mathbb{E}_{\pi} \left[ r + V[s'] \ \vert \ s_t = s \right]$$ is the same as taking the maximum over all actions of $$\mathbb{E}_{\pi} \left[ r + V[s'] \ \vert \ s_t = s, a_t = a \right]$$ (because all of the $$\pi(a \ \vert \ s)$$ terms drop out of the prior except for the the term where $$a = a_{\text{max}}$$. This implies that we can combine the policy evaluation and policy improvement steps into a single update,

$$V_{k + 1}[s] = \argmax_{a} \mathbb{E}_{\pi} \left[ r + V_k[s'] \ \vert \ s_t = s, a_t = a \right],$$

which turn out to be exactly the update we would get by translating the Bellman *optimality* equation into an update rule. The resulting algorithm (demonstrated below) is known as the *Value Iteration* algorithm.

```julia
# Algorithm (Value Iteration)
function value_iteration(pR, pS, R, gamma; tol=1e-6)
    nstates, nactions, _ = size(pS)
    V = zeros(nstates)
    policy = zeros(nstates, nactions)  # policy[s, a] = π(a | s)
    Δ = Inf
    i = 0
    while Δ > tol
        Δ = 0.0
        for (v_idx, v) in enumerate(V)
            a_max = 0
            Q_max = -Inf
            for a_idx in 1:nactions
                Q = 0.0
                for (s_idx, ps) in enumerate(pS[v_idx, a_idx, :])
                    v_next = V[s_idx]
                    for (r_idx, pr) in enumerate(pR[v_idx, a_idx, :])
                        r = R[r_idx]
                        Q += ps * pr * (r + v_next)
                    end
                end
                if Q > Q_max
                    a_max = a_idx
                    Q_max = Q
                end
            end
            V[v_idx] = Q_max
            policy[v_idx, a_max] = 1.0
            Δ = max(Δ, abs(V[v_idx] - v))
        end
        i += 1
    end
    println("Converged in $i steps (Δ = $Δ).")
    return policy, value
end
```



What's required to perform these operations? We need to know the policy ($$\pi(a \ \vert \ s)$$), and we need to know the *complete* dynamics of the system ($$ p(s', r \ \vert \ s, a) $$)—or at least be able to make accurate estimates of the transition probabilities.

### Extensions
- Importance weighting updates?



## Monte Carlo
