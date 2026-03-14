# Cartpole AI — Deep Q-Network Agent

A DQN (Deep Q-Network) agent that learns to balance a pole on a cart using experience replay and an epsilon-greedy exploration strategy.

## What It Does

The agent interacts with the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment from Gymnasium. A small feedforward neural network approximates Q-values, and the agent trains online using randomly sampled minibatches from a replay buffer.

**Architecture:**
- 2 hidden layers (24 units each, ReLU)
- Linear output layer (one Q-value per action)
- MSE loss, Adam optimizer (lr = 0.001)
- Epsilon-greedy exploration decaying from 1.0 → 0.01

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| 🧠 RL Environment | Gymnasium (CartPole-v1) |
| 🔧 Neural Network | TensorFlow / Keras |
| 📊 Numerics | NumPy |
| 🐍 Language | Python 3.9+ |

## Getting Started

```bash
pip install -r requirements.txt
python dqn_keras.py
```

The agent trains for 1000 episodes by default. Scores are printed each episode.

## ⚠️ Known Issues

- Training is slow because each sample in the minibatch triggers a separate `model.fit` call. A vectorised batch update would be significantly faster.
- No model saving/loading — trained weights are lost when the script exits.
- `CartPole-v1` caps episodes at 500 steps; the agent typically solves it well before that.
