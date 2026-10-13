import gymnasium as gym
import random

env = gym.make("CartPole-v1")
state, _ = env.reset()

for step in range(200):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()

env.close()
