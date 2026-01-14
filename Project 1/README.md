## Overview
This project investigates and compares two reinforcement learning algorithms — Deep Q-Network (DQN) and Advantage Actor–Critic (A2C) — on the CartPole-v1 environment.

The main objective is to analyze how different reinforcement learning paradigms behave under:
- different reward designs,
- hyperparameter settings,
- and random seeds.

The project was developed as part of a university-level Reinforcement Learning course.

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
- Double DQN (optional)

---

### Advantage Actor–Critic (A2C)
An actor–critic method combining policy-based and value-based learning.

Key components:
- Shared actor–critic neural network  
- n-step rollouts  
- Entropy regularization for exploration  
- Bootstrapped value estimates  

The two models were intentionally chosen from different RL paradigms to enable a meaningful comparison.

---

## Reward Functions

### Baseline Reward
The default CartPole environment reward.

Characteristics:
- Sparse feedback
- Weak learning signal

Performance with baseline reward:
- DQN: evaluation mean ≈ 103  
- A2C: evaluation mean ≈ 10  

With the baseline reward, both models fail to learn an effective control policy.

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

Effect on performance:
- Both DQN and A2C show significantly improved learning behavior
- Enables the agents to reach near-optimal performance

Reward shaping proved essential for successful training in this environment.

---

## Hyperparameter Tuning

A limited but systematic grid search was performed.

### DQN Search Space (12 configurations)
- Learning rate: `{1e-4, 5e-4, 1e-3}`
- Discount factor (γ): `{0.95, 0.99}`
- Target network update: `{500, 1000}`

### A2C Search Space (48 configurations)
- Learning rate: `{1e-3, 5e-4, 2e-4}`
- Discount factor (γ): `{0.97, 0.99}`
- Hidden units: `{64, 128}`
- n-step rollouts: `{5, 10}`
- Entropy coefficient: `{0.0, 0.01}`

Each configuration was evaluated using multiple episodes.

---

## Results

### Best Single-Seed Performance (Shaped Reward)

| Model | Mean Return | Std | Solved |
|------|------------|-----|--------|
| DQN  | ≈ 481 | ≈ 29.7 | Yes |
| A2C  | ≈ 496 | ≈ 0.6  | Yes |

Both algorithms successfully solved the CartPole environment under favorable conditions.

Learning curves, evaluation statistics, and video rollouts are included in the repository.

---

## Evaluation Strategy
- Evaluation over 100 episodes
- Metrics: mean return, standard deviation
- A2C evaluated in two modes:
  - Greedy (argmax action)
  - Stochastic (sampling from policy)

This allows assessment of both deterministic performance and robustness.

---

## Multi-Seed Robustness Analysis

To evaluate stability, the best configurations were trained and evaluated across **3 different random seeds**.

### Final Results (Mean ± Std Across Seeds)

| Model | Mean Return |
|------|------------|
| DQN | 348.8 ± 59.0 |
| A2C (greedy) | 405.1 ± 65.2 |
| A2C (stochastic) | 402.4 ± 65.9 |

Although both models can solve CartPole in single runs, multi-seed evaluation reveals sensitivity to random initialization.

---

## Challenges and Solutions

### Challenges
- Sparse rewards
- Training instability
- High variance in A2C
- Sensitivity to random seeds

### Solutions
- Reward shaping
- Hyperparameter tuning
- Gradient clipping
- Multi-seed evaluation
- Smoothed learning curves

---

## Conclusions

- Reward shaping is critical for effective learning in CartPole.
- Both DQN and A2C can solve the environment under favorable conditions.
- A2C converges faster but exhibits higher variance.
- DQN is more stable but learns more slowly.
- Single-seed success does not guarantee robustness.
- Multi-seed evaluation is essential for reliable conclusions in reinforcement learning.

---
