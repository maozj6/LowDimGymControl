import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import random
from collections import deque

ENV_ID = "LunarLanderContinuous-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SEED = 42
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
TOTAL_STEPS = 500_000
START_STEPS = 10_000
UPDATE_AFTER = 10_000
UPDATE_EVERY = 50

LOG_STD_MIN = -20
LOG_STD_MAX = 2

SAVE_DIR = "sac_lander_ckpt"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# Utils
# ----------------------
def set_seed(env, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

# ----------------------
# Replay Buffer
# ----------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def add(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        o, a, r, o2, d = map(np.stack, zip(*batch))
        return o, a, r, o2, d

    def __len__(self):
        return len(self.buf)

# ----------------------
# Networks
# ----------------------
def mlp(input_dim, layers, output_dim, act=nn.ReLU):
    modules = []
    prev = input_dim
    for h in layers:
        modules += [nn.Linear(prev, h), act()]
        prev = h
    modules.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*modules)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim*2)

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, logstd = mu_logstd.chunk(2, dim=-1)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        return mu, logstd

    def sample(self, obs):
        mu, logstd = self(obs)
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)

        # logprob correction
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=[256,256]):
        super().__init__()
        self.q = mlp(obs_dim+act_dim, hidden, 1)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))

# ----------------------
# SAC Agent
# ----------------------
class SAC:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic1 = Critic(obs_dim, act_dim).to(DEVICE)
        self.critic2 = Critic(obs_dim, act_dim).to(DEVICE)
        self.tgt_c1 = Critic(obs_dim, act_dim).to(DEVICE)
        self.tgt_c2 = Critic(obs_dim, act_dim).to(DEVICE)

        self.tgt_c1.load_state_dict(self.critic1.state_dict())
        self.tgt_c2.load_state_dict(self.critic2.state_dict())

        self.entropy_coef = 0.2
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=LR
        )

    def select_action(self, obs, eval=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if eval:
            with torch.no_grad():
                mu, _ = self.actor(obs_t)
                a = torch.tanh(mu)
            return a.cpu().numpy()[0]
        else:
            with torch.no_grad():
                a, _ = self.actor.sample(obs_t)
            return a.cpu().numpy()[0]

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def update(self, buffer):
        o, a, r, o2, d = buffer.sample(BATCH_SIZE)
        o=torch.as_tensor(o, dtype=torch.float32, device=DEVICE)
        a=torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
        r=torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        o2=torch.as_tensor(o2, dtype=torch.float32, device=DEVICE)
        d=torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Next action
        a2, logp2 = self.actor.sample(o2)

        with torch.no_grad():
            q1_tgt = self.tgt_c1(o2, a2)
            q2_tgt = self.tgt_c2(o2, a2)
            q_tgt = torch.min(q1_tgt, q2_tgt) - self.entropy_coef * logp2
            backup = r + GAMMA * (1-d) * q_tgt

        # Critic loss
        q1 = self.critic1(o,a)
        q2 = self.critic2(o,a)
        critic_loss = ((q1-backup)**2 + (q2-backup)**2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor loss
        a_new, logp = self.actor.sample(o)
        q1_new = self.critic1(o, a_new)
        q2_new = self.critic2(o, a_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.entropy_coef * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Target networks
        self.soft_update(self.critic1, self.tgt_c1)
        self.soft_update(self.critic2, self.tgt_c2)
def main():
    env = gym.make(ENV_ID)
    set_seed(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SAC(obs_dim, act_dim)
    buffer = ReplayBuffer(BUFFER_SIZE)

    o = env.reset()
    ep_ret = 0
    ep_len = 0
    episode_rewards = []
    recent_rewards = deque(maxlen=20)  # 最近20回合均值

    actor_loss = None
    critic_loss = None

    for step in range(1, TOTAL_STEPS+1):

        # random explore at beginning
        if step < START_STEPS:
            a = env.action_space.sample()
        else:
            a = agent.select_action(o)

        o2, r, done, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        buffer.add(o, a, r, o2, float(done))
        o = o2

        if done:
            episode_rewards.append(ep_ret)
            recent_rewards.append(ep_ret)
            print(f"[Episode End] Steps: {step} | Return: {ep_ret:.1f} | Last 20 Avg: {np.mean(recent_rewards):.1f}")

            o = env.reset()
            ep_ret = 0
            ep_len = 0

        # SAC update
        if step > UPDATE_AFTER and step % UPDATE_EVERY == 0:
            for _ in range(UPDATE_EVERY):
                o_b, a_b, r_b, o2_b, d_b = buffer.sample(BATCH_SIZE)
                # ——Reuse update code——
                # small change: return losses to log
                # Critic
                o_t = torch.as_tensor(o_b, dtype=torch.float32, device=DEVICE)
                a_t = torch.as_tensor(a_b, dtype=torch.float32, device=DEVICE)
                r_t = torch.as_tensor(r_b, dtype=torch.float32, device=DEVICE).unsqueeze(1)
                o2_t = torch.as_tensor(o2_b, dtype=torch.float32, device=DEVICE)
                d_t = torch.as_tensor(d_b, dtype=torch.float32, device=DEVICE).unsqueeze(1)

                a2, logp2 = agent.actor.sample(o2_t)
                with torch.no_grad():
                    q1_tgt = agent.tgt_c1(o2_t, a2)
                    q2_tgt = agent.tgt_c2(o2_t, a2)
                    q_tgt = torch.min(q1_tgt, q2_tgt) - agent.entropy_coef * logp2
                    backup = r_t + GAMMA * (1-d_t) * q_tgt

                q1 = agent.critic1(o_t, a_t)
                q2 = agent.critic2(o_t, a_t)
                critic_loss = ((q1-backup)**2 + (q2-backup)**2).mean()

                agent.critic_opt.zero_grad()
                critic_loss.backward()
                agent.critic_opt.step()

                a_new, logp_new = agent.actor.sample(o_t)
                q1_new = agent.critic1(o_t, a_new)
                q2_new = agent.critic2(o_t, a_new)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (agent.entropy_coef * logp_new - q_new).mean()

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()

                agent.soft_update(agent.critic1, agent.tgt_c1)
                agent.soft_update(agent.critic2, agent.tgt_c2)

        # 训练状态日志
        # 训练状态日志
        if step % 10000 == 0:
            avg20 = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0

            if actor_loss is None or critic_loss is None:
                aloss = "N/A"
                closs = "N/A"
            else:
                aloss = f"{actor_loss:.3f}"
                closs = f"{critic_loss:.3f}"

            print(f"[{step}/{TOTAL_STEPS}] "
                  f"Last20Avg={avg20:.1f} | "
                  f"ActorLoss={aloss} | CriticLoss={closs}")

        # 定期保存模型
        if step % 10000 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(SAVE_DIR, f"actor_{step}.pt"))
            print(f"Model saved at step {step}")

    env.close()
    torch.save(agent.actor.state_dict(), os.path.join(SAVE_DIR, "actor_final.pt"))
    print("Training finished!")

if __name__ == "__main__":
    main()
