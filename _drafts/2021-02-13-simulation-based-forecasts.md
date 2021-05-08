---
layout: post
author: Colin Swaney
title: Simulation-Based Forecasts
date: 2021-02-13
categories: [research]
category: research
tags: [research]
excerpt: "<p></p>"
---

## Bayesian Inference
- You can do everything with this but...
- Generally simplify the model to apply inference methods...
- But new flexibility from Turing.jl, etc.?

## Simulation-Based Inference
- What you can do:

- What you can't (easily) do:
  - Parameter inference
    - No Posterior distribution (can Turing.jl, etc. do this?)
    - Maybe asymptotic MLE distribution?

## Example: State-Space Model
Here is an interesting example. Interesting because it demonstrates that relatively complex statistical methodology can be approximated by simpler, but highly detailed, simulation model, and vice-a-versa.

The statistical model we have in mind is a state-space model, but let's give it some interpretation for fun. So imagine we are trying to model election outcomes based soley on polling data. Polls are imperfect measurements of the true percent of voters that will vote for a given candidate at a given time. That percent can and will change over time. Our goal is to forecast what the percent will be when the election arrives.

### Classic Approach
Let's write down a model.

First, the true percent of voters that will vote for a given candidate at any time (what we're modeling) is a time series. We could give it any dynamics we want, but let's say for simplicity that it just bounces around with normal errors:

$$ \rho_t = \rho_{t - 1} + \epsilon_t, \ \  \epsilon_t \sim N(0, \sigma_p^2) $$

Next, polls are imperfect observations of the true probability. In reality, we would observe *many* polls, all of which represent imperfect information about the true percent of voters. But let's simplify life by assuming we have access to a consolidated poll. It varies around the true percent of voters with normal errors as well:

$$ s_t = \rho_t + \eta_t, \ \ \eta_t \sim N(0, \sigma_s^2) $$

Now our goal is to forecast the percent of voters on election day, $$\rho_T$$. With our model, the current value of $$\rho_t$$ is also the most likely value on election day, but we're interesting in a distribution of $$\rho_T$$. If we *know* what the current percent is, then all we need to do is simulate $$\rho_t$$ forward to see what that distribution is. So we really only need to estimate the current percent.

This model is actually a classical model. Formally, we want to estimate the percent of voters conditional on all polling data observed up to and including the current time, $$p(\rho_t \vert s_{1:t})$$. There is a standard technique for estimating this value known as Kalman filtering. The key idea to is to start with beliefs about the value of $$\rho_t$$, and then iterative update those beliefs as information is revealed through $$s_t$$. 

Working in this manner, beliefs at any time $$t$$ incorporate all information up to that time and *adjust for the staleness of infomration*. That is, our current estimate of accounts for the fact that we saw $$s_{t - \Delta t}$$, but if $$\Delta t$$ is large, then its impact on $$\hat{p}_t$$ is small.

### Simulation Approach
- There isn't much to do here... all the (initial) work goes to pooling information from polls intelligently.
- Then you can simulate past information...
- Then you can simulate future evolution...
- Add any details you like really...

As an slight aside, the standard deviation of the the forecast scales with the square root of the days between now and the election. For example, if the election is 100 days out, then the standard deviation of $$\rho_T$$ is ten times that of $$\rho_t$$. As a result, the effect of new information of our *forecast* is considerable smaller than its effect on our current estimate[^1].

[^1]: This seems to be Naseem Talib's critique of the FiveThirtyEight prediction model.

![Forecast](/assets/img/forecast.pdf)




Practical challenges to think about:
1. How would we actually handle multiple polls?
2. How would we deal with polls arriving on different days?
