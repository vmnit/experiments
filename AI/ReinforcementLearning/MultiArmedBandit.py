import random

# Environment
class Bandit:
    def __init__(self, probs):
        self.probs = probs  # success probability for each arm

    def pull(self, arm):
        return 1 if random.random() < self.probs[arm] else 0


# Agent
class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = [0.0] * n_actions
        self.action_counts = [0] * n_actions

    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.q_values) - 1)
        return self.q_values.index(max(self.q_values))

    def update(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


# Training loop
bandit = Bandit(probs=[0.2, 0.5, 0.75])
agent = EpsilonGreedyAgent(n_actions=3, epsilon=0.1)

total_reward = 0
for step in range(1000):
    action = agent.select_action()
    reward = bandit.pull(action)
    agent.update(action, reward)
    total_reward += reward

print("Estimated action values:", agent.q_values)
print("Total reward:", total_reward)
