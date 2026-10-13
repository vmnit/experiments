import random

# Grid parameters
GRID_SIZE = 5
ACTIONS = ["up", "down", "left", "right"]

# Q-table
Q = {}

def get_q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: get_q(state, a))

def step(state, action):
    x, y = state

    if action == "up":
        x = max(0, x - 1)
    elif action == "down":
        x = min(GRID_SIZE - 1, x + 1)
    elif action == "left":
        y = max(0, y - 1)
    elif action == "right":
        y = min(GRID_SIZE - 1, y + 1)

    new_state = (x, y)
    reward = 10 if new_state == (4, 4) else -1
    done = new_state == (4, 4)

    return new_state, reward, done


# Training
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor

for episode in range(500):
    state = (0, 0)
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = step(state, action)

        best_next_q = max(get_q(next_state, a) for a in ACTIONS)
        old_q = get_q(state, action)

        Q[(state, action)] = old_q + alpha * (
            reward + gamma * best_next_q - old_q
        )

        state = next_state

print("Training completed")

state = (0, 0)
path = [state]

while state != (4, 4):
    action = choose_action(state, epsilon=0.0)
    state, _, _ = step(state, action)
    path.append(state)

print("Optimal path:", path)
