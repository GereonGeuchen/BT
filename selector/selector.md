# Selector Architecture and Evaluation

Our approach involves a hierarchical decision process for selecting the best algorithm and switching point under a fixed evaluation budget. The goal is to maximize final precision by strategically choosing when to switch algorithms and which algorithm to switch to.

## Algorithm Space

We define the algorithm space as **A × S**, where:
- **A**=$\{\text{BFGS}, \text{DE}, \text{PSO}, \text{MLSL}, \text{Same}, \text{Non-elitist}\}$ is the set of algorithms,
- **S**=$\{50, 100, 150, \ldots,1000\}$ is the set of discrete **switching points**

Each algorithm selection is therefore specific to a switching point, and the optimal policy is a joint function of both.

## Approach Overview

Let **n** be the total number of switching points. Our selector consists of two components per switching point:

1. **Switching Decision Regressors**:
   - For each switching point $ s_i $, we train $ n - (i + 1) $ regression models, where $ n $ is the total number of switching points.
   - Each model predicts the optimal precision achievable at a later switching point $ s_j $, where $ j > i $, assuming perfect algorithm choice at $ s_j $.
   - We compare the predicted optimal performance of future points with the current switching point's predicted precision.  
     **If the current point is expected to yield the best performance, we decide to switch.**

2. **Algorithm Performance Regressors**:
   - For each switching point $ s_i $ we train one regression model **per algorithm** (typically 6).
   - These models predict the algorithm-specific precision at $ s_i $, and the best-performing algorithm is selected when switching occurs.

> **Note:** We do not train either the switching-point regressors or the algorithm regressors for the final switching point $ s_{n} $.  
> Reaching this point means the selector has opted to never switch — i.e., defer the decision until the budget is fully consumed.

## Procedure

1. **Train Switching Performance Predictors**:
   - For each switching point $ s_i $, train regressors to predict the optimal precision at all later switching points $ s_{i+1}, \ldots, s_{n} $.

2. **Train Algorithm Performance Predictors**:
   - For each switching point $ s_i $, train a regression model for each algorithm $ a \in A $ to predict its performance at $ s_i $.

3. **Execute Switching Strategy**:
   - For a given instance, evaluate all $ s_i \in S $ sequentially.
   - At each $ s_i $, check if the current point is predicted to yield the best future performance.
   - If yes, query the algorithm regressors at $ s_i $ to select the best algorithm and finalize the choice.
   - If not, continue evaluating later switching points.

4. **Evaluate Performance**:
   - Measure the final precision achieved by this switching + selection pipeline.
   - Compare against the **Virtual Best Solver (VBS)**, defined as the selector that always picks the best algorithm in hindsight at the best switching point (as per Hadar's definition).

## Summary

This approach captures both:
- The **temporal decision** of *when* to switch
- The **algorithmic decision** of *what* to switch to

By learning from precision predictions across the A × S space, the system balances exploration of future gains with immediate switching decisions.
