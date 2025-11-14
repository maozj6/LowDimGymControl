import gym
import torch
import numpy as np
from torch import nn

BEST_MODEL_PATH = "best_score_model.pth"
TEST_EPISODES = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Model Structure (must match training)
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

# Load env and model
env = gym.make("CartPole-v1")
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
policy_net.eval()

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy_net(state).argmax(1).item()

def parse_step(result):
    """ Compatible parsing for gym / gymnasium """
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return next_state, reward, done

reward_list = []

print("======== Testing Agent ========")

for ep in range(TEST_EPISODES):
    state = env.reset()
    if isinstance(state, tuple):
        state, _ = state  # unwrap

    total_reward = 0
    while True:
        action = select_action(state)
        result = env.step(action)
        next_state, reward, done = parse_step(result)
        total_reward += reward
        state = next_state
        if done:
            break

    reward_list.append(total_reward)
    print(f"Episode {ep:3d} | Reward = {total_reward}")

env.close()

avg_reward = np.mean(reward_list)
std_reward = np.std(reward_list)

print("======= Evaluation Done ✅ =======")
print(f"Average Reward over {TEST_EPISODES} episodes: {avg_reward:.2f}")
print(f"Std Dev: {std_reward:.2f}")
