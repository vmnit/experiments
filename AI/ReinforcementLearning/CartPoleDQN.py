import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# ---------------------------
# Hyperparameters
# ---------------------------
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
EPISODES = 300

device = torch.device("cpu")

# ---------------------------
# Q-Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s2),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# Training Setup
# ---------------------------
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

steps_done = 0

def epsilon_by_step(step):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)

# ---------------------------
# Training Loop
# ---------------------------
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        steps_done += 1
        epsilon = epsilon_by_step(steps_done)

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            s, a, r, s2, d = memory.sample(BATCH_SIZE)

            q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = target_net(s2).max(1)[0]
                target = r + GAMMA * next_q * (1 - d)

            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward:.0f}, Epsilon: {epsilon:.3f}")

env.close()
