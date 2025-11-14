import gym
import torch
import numpy as np
import time
from train import Actor, DEVICE, ENV_ID

CKPT = r"E:\25Nov\decomp\lunar\sac_lander_ckpt\actor_120000.pt"
TEST_EPISODES = 1
MAX_STEPS = 500   # 给飞船落地点时间

def run_test():
    env = gym.make(ENV_ID)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(DEVICE)
    actor.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    actor.eval()

    rewards = []

    for ep in range(TEST_EPISODES):
        o = env.reset()
        # ✅ 兼容 tuple 格式 (新gym) 和 array 格式 (旧gym)
        if isinstance(o, tuple):
            o = o[0]

        total_r = 0.0

        for step in range(MAX_STEPS):
            env.render()  # ✅ 正确方式
            time.sleep(0.03)

            obs = torch.as_tensor(o, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(obs)
                a = torch.tanh(mu).cpu().numpy()[0]

            o, r, done, info = env.step(a)
            total_r += r

            if done:
                break

        rewards.append(total_r)
        print(f"Episode {ep+1} | Reward: {total_r:.2f} | Steps: {step+1}")

    print("\n=============================")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print("=============================")

    env.close()


if __name__ == "__main__":
    run_test()
