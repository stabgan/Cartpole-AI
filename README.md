# 🎮 CartPole AI — Deep Q-Network Agent

A Deep Q-Learning (DQN) agent that learns to balance a pole on a cart using the classic CartPole environment. The agent uses a neural network to approximate Q-values and an experience replay buffer to stabilize training.

## 🧠 How It Works

The agent observes the cart's position, velocity, pole angle, and angular velocity, then decides whether to push the cart left or right. Over many episodes it learns an optimal policy through trial and error, gradually shifting from random exploration to learned behavior via an epsilon-greedy strategy.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🏋️ RL Environment | [Gymnasium](https://gymnasium.farama.org/) (CartPole-v1) |
| 🧬 Neural Network | [Keras](https://keras.io/) (Sequential model) |
| 🔢 Numerical Computing | [NumPy](https://numpy.org/) |
| 🐍 Language | Python 3.8+ |

## 📦 Dependencies

```
gymnasium
numpy
keras
tensorflow
```

Install everything at once:

```bash
pip install gymnasium numpy keras tensorflow
```

## 🚀 How to Run

```bash
python dqn_keras.py
```

The agent will train for 1000 episodes. You'll see output like:

```
episode: 0/1000, score: 14
episode: 1/1000, score: 22
...
episode: 999/1000, score: 499
```

To enable visual rendering, uncomment the `env.render()` line in `dqn_keras.py` and create the environment with render mode:

```python
env = gym.make('CartPole-v1', render_mode='human')
```

## ⚠️ Known Issues

- Training is slow since predictions and fitting happen per-sample inside the replay loop (no batch matrix ops).
- No model saving/loading — trained weights are lost when the script exits.
- No target network — using a single network for both prediction and target computation can cause instability.
- The replay buffer must fill to at least `batch_size` (32) samples before training begins.
