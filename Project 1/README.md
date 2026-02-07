# Reinforcement Learning on CartPole-v1  
**DQN vs A2C with Baseline and Shaped Rewards**

---

## Overview
This project investigates and compares two reinforcement learning algorithms — **Deep Q-Network (DQN)** and **Advantage Actor–Critic (A2C)** — on the **CartPole-v1** environment.

The main objective is to analyze how different reinforcement learning paradigms behave under:
- different reward designs,
- hyperparameter settings,
- and random seeds.

The project was developed as part of a **university-level Reinforcement Learning course**.

---

## Environment
**CartPole-v1**

- Observation space: 4-dimensional continuous state  
- Action space: 2 discrete actions  
- Maximum episode length: 500 steps  
- Solved criterion:  
  **Average return ≥ 475 over 100 evaluation episodes**

CartPole was selected because it is:
- computationally efficient,
- a standard RL benchmark,
- well-suited for algorithmic comparison.

---

## Implemented Algorithms

### Deep Q-Network (DQN)
A value-based reinforcement learning method.

Key components:
- Experience replay buffer  
- Target network  
- ε-greedy exploration  
- Double DQN  

---

### Advantage Actor–Critic (A2C)
An actor–critic method combining policy-based and value-based learning.

Key components:
- Shared actor–critic neural network  
- n-step rollouts  
- Entropy regularization for exploration  
- Bootstrapped value estimates  

The two models were intentionally chosen from different reinforcement learning paradigms to enable a meaningful comparison.

---

## Reward Functions

### Baseline Reward
The default CartPole environment reward.

Characteristics:
- Sparse feedback
- Weak learning signal for policy-gradient methods

---

### Shaped Reward (shaped_v2)
A custom reward function designed to provide more informative feedback during training.

Description:
- Penalizes large deviations of the pole angle from the upright position
- Encourages stability and balanced pole control

Key properties:
- More informative learning signal
- Faster convergence
- Improved training stability

---

## Hyperparameter Tuning

A limited but systematic grid search was performed.

### DQN Search Space (12 configurations)
- Learning rate: `{1e-4, 5e-4, 1e-3}`
- Discount factor (γ): `{0.95, 0.99}`
- Target network update frequency: `{500, 1000}`

**Best DQN configuration (shaped reward):**
- lr = `5e-4`
- γ = `0.95`
- target update = `1000`

---

### A2C Search Space (48 configurations)
- Learning rate: `{1e-3, 5e-4, 2e-4}`
- Discount factor (γ): `{0.97, 0.99}`
- Hidden units: `{64, 128}`
- n-step rollouts: `{5, 10}`
- Entropy coefficient: `{0.0, 0.01}`

**Best A2C configuration (shaped reward):**
- lr = `5e-4`
- γ = `0.97`
- hidden = `128`
- n_steps = `5`
- entropy_coef = `0.0`

---

## Results

### Baseline Reward — Single-Seed Results

| Model | Train Episodes | Eval Episodes | Mean Return | Std | Solved |
|------|----------------|---------------|-------------|-----|--------|
| DQN  | 800  | 200 | **500.0** | **0.0** | Yes |
| A2C  | 1200 | 200 | **16.3**  | 2.3 | No |

With the baseline environment reward, **DQN successfully learns an optimal policy** and consistently solves the CartPole task.  
In contrast, **A2C fails to learn a stable control policy**, remaining close to random performance even with extended training.

---

### Shaped Reward (shaped_v2) — Single-Seed Results

| Model | Train Episodes | Eval Episodes | Mean Return | Std | Solved |
|------|----------------|---------------|-------------|-----|--------|
| DQN  | 600 | 200 | 470.8 | 2.7 | No |
| A2C  | 900 | 200 | **496.1** | **0.3** | Yes |

Under the shaped reward, both algorithms improve substantially.  
**A2C achieves near-perfect and highly stable performance**, while **DQN improves learning speed but remains slightly below the solving threshold** in the final single-seed run.

Learning curves, evaluation statistics, and video rollouts are included in the repository.

---

## Evaluation Strategy
- Evaluation over **200 episodes**
- Metrics: mean return, standard deviation
- A2C evaluated in two modes:
  - **Greedy** (argmax action)
  - **Stochastic** (sampling from policy)

This allows assessment of both deterministic performance and robustness.

---

## Multi-Seed Robustness Analysis

To evaluate stability and reproducibility, the best-performing **shaped-reward** configurations were trained and evaluated across **3 different random seeds**.

### Shaped Reward — Multi-Seed Results (Mean ± Std)

| Model | Mean Return |
|------|------------|
| DQN | **399.4 ± 69.2** |
| A2C (greedy) | **467.6 ± 40.6** |
| A2C (stochastic) | **464.6 ± 43.6** |

Although both algorithms can solve CartPole under favorable single-seed conditions, multi-seed evaluation reveals substantial sensitivity to random initialization.  
In particular, **DQN exhibits high variance across seeds**, while **A2C demonstrates more consistent performance**.

---

## Challenges and Solutions

### Challenges
- Sparse rewards
- Training instability
- Sensitivity to random seeds
- High variance in policy-gradient methods

### Solutions
- Reward shaping
- Hyperparameter tuning
- Gradient clipping
- Multi-seed evaluation
- Smoothed learning curves

---

## Conclusions

- **Baseline reward**
  - DQN reliably solves CartPole.
  - A2C fails to learn effectively.

- **Shaped reward**
  - Dramatically improves A2C performance and stability.
  - Improves DQN learning speed but introduces higher variance.

- **Robustness**
  - Single-seed success does not guarantee reliability.
  - Multi-seed evaluation is essential for fair reinforcement learning comparison.

Overall, this study highlights the strong interaction between **algorithm choice**, **reward design**, and **training stability** in reinforcement learning.
