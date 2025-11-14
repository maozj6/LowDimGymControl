import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Config
EPISODES = 2000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = 50000

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_TAU = 0.005  # Soft update

SAVE_PATH_BEST = "best_score_model.pth"
SAVE_PATH_LATEST = "latest_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)


env = gym.make("CartPole-v1")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START
best_avg = -float("inf")


def soft_update():
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TARGET_UPDATE_TAU * param.data +
                                (1 - TARGET_UPDATE_TAU) * target_param.data)


def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, 1)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(state).argmax(1).item()


def replay():
    if len(memory) < BATCH_SIZE:
        return None

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target = rewards + GAMMA * next_q_values * (1 - dones)

    # Huber loss (more stable than MSE)
    loss = nn.SmoothL1Loss()(q_values.squeeze(), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    soft_update()

    return loss.item()


scores = []
for episode in range(EPISODES):
    state = env.reset()
    if isinstance(state, tuple):  # old gym compatibility
        state = state[0]

    total_reward = 0
    loss = None

    for _ in range(1000):
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        loss = replay()

        if done:
            break

    # epsilon decay
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    scores.append(total_reward)
    avg = np.mean(scores[-50:])  # moving average
    print(f"Episode {episode:4d} | Score: {total_reward:6.1f} | Avg50: {avg:6.1f} | Epsilon: {epsilon:.3f}")

    # 保存最好模型
    if avg > best_avg:
        best_avg = avg
        torch.save(policy_net.state_dict(), SAVE_PATH_BEST)

    # 保存最新模型
    torch.save(policy_net.state_dict(), SAVE_PATH_LATEST)

env.close()
print("Training complete ✅")
print(f"Best model saved to {SAVE_PATH_BEST}")
print(f"Latest model saved to {SAVE_PATH_LATEST}")
