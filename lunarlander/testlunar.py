import gym
import torch
import numpy as np
from train import Actor, DEVICE, ENV_ID   # 导入你的模型定义

CKPT = "E:\\25Nov\decomp\lunar\sac_lander_ckpt//actor_140000.pt"  # 改成你要测试的ckpt路径
TEST_EPISODES = 1000
MAX_STEPS = 500
# 60k,100,61.20
# 70k,100,68.20
#80k,100,67.72
# Average Reward over 100 eps = 75.28
# Average Reward over 100 eps = 84.13
# Average Reward over 100 eps = 77.25
# Average Reward over 100 eps = 74.88


# Average Reward over 100 eps = 253.49
# Average Reward over 100 eps = 263.06
# Average Reward over 100 eps = 272.78

def run_test():
    env = gym.make(ENV_ID)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 加载模型
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    actor.eval()

    rewards = []

    for ep in range(TEST_EPISODES):
        o = env.reset()
        done = False
        total_r = 0.0

        for _ in range(MAX_STEPS):
            obs = torch.as_tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(obs)
                a = torch.tanh(mu).cpu().numpy()[0]
            o, r, done, _ = env.step(a)
            total_r += r
            if done:
                break

        rewards.append(total_r)
        print(f"Episode {ep+1}: Reward = {total_r:.2f}")

    print("\n==============================")
    print(f"Average Reward over {TEST_EPISODES} eps = {np.mean(rewards):.2f}")
    print("==============================")

    env.close()


if __name__ == "__main__":
    run_test()
