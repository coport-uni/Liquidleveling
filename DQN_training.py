"""
=============================================================================
수위 제어 시스템 - DQN 강화학습 (통합본 / 소문자 변수명 / done 제거)
- action = 절대 펌프값 u ∈ {5..15} (정수), 총 11개 액션
- 상태에서 time 제거
- continuing task 가정으로 transition에서 done 제거
=============================================================================
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt


# =============================================================================
# 1. 시스템 설정 (모두 소문자)
# =============================================================================

tank_height_cm = 8.0
setpoint_cm = 3.85
control_period_s = 3.0

# 물리적으로 가능한 펌프 범위(참고용)
phys_min_pump = 0
phys_max_pump = 20

# 학습/제어에서 사용할 펌프 범위 (정수)
act_u_min = 5
act_u_max = 15
num_actions = act_u_max - act_u_min + 1  # 11

# 상태 / 액션 차원
state_dim = 5   # [h, h_prev, error, error_int, u_prev]
action_dim = num_actions  # 11

# dqn 파라미터
learning_rate = 1e-4
gamma = 0.99
tau = 0.005

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

buffer_size = 100000
batch_size = 64
min_buffer_size = 1000

max_steps_per_episode = 100
num_episodes = 1000

safety_h_min = 1.0
safety_h_max = 7.0


# =============================================================================
# 2. Replay Buffer (done 제거)
# =============================================================================

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append(Transition(state, action, reward, next_state))

    def sample(self, n: int):
        batch = random.sample(self.buffer, n)
        batch = Transition(*zip(*batch))
        states = torch.tensor(np.array(batch.state), dtype=torch.float32)
        actions = torch.tensor(batch.action, dtype=torch.int64)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 3. DQN Network
# =============================================================================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim),
        )

        # xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4. DQN Agent (time 제거, 11개 action: u=5..15, done 제거)
# =============================================================================

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.steps_done = 0
        self.episodes_done = 0

        self.h_prev = setpoint_cm
        self.u_prev = float((act_u_min + act_u_max) // 2)  # 10
        self.error_int = 0.0

    # action(0..10) -> u(5..15)
    def action_to_u(self, action: int) -> int:
        return int(act_u_min + action)

    def u_to_action(self, u: int) -> int:
        return int(u - act_u_min)

    def reset_episode(self, h0: float):
        self.h_prev = float(h0)
        self.u_prev = float((act_u_min + act_u_max) // 2)
        self.error_int = 0.0

    def build_state(self, h: float) -> np.ndarray:
        error = setpoint_cm - h
        return np.array([
            h / tank_height_cm,
            self.h_prev / tank_height_cm,
            error / tank_height_cm,
            np.clip(self.error_int, -10.0, 10.0) / 10.0,
            self.u_prev / phys_max_pump,
        ], dtype=np.float32)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, action_dim - 1)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q = self.policy_net(s)[0]
            return int(torch.argmax(q).item())

    def compute_reward(self, h: float, u: float) -> float:
        # 1) tracking penalty
        error = abs(h - setpoint_cm)
        tracking = -10.0 * (error ** 2)

        # 2) smoothness penalty
        du = abs(u - self.u_prev)
        smooth = -0.1 * (du ** 2)

        # 3) safety penalty
        safety = 0.0
        if h > safety_h_max:
            safety = -100.0
        elif h < safety_h_min:
            safety = -50.0

        # 4) bonus
        bonus = 5.0 if error < 0.1 else 0.0

        return float(tracking + smooth + safety + bonus)

    def train_step(self):
        if len(self.buffer) < min_buffer_size:
            return None

        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Q(s,a)
        q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # target = r + gamma * max Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            target = rewards + gamma * q_next

        loss = nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(tau * p.data + (1 - tau) * t.data)

    def decay_epsilon(self):
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
        }, path)
        print(f"model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt["epsilon"])
        self.episodes_done = int(ckpt.get("episodes_done", 0))
        print(f"model loaded from {path}")


# =============================================================================
# 5. Simulator (done 제거)
# =============================================================================

class WaterTankSimulator:
    def __init__(self):
        self.a, self.b, self.c = 0.95, 0.05, -0.1
        self.h = setpoint_cm
        self.step_count = 0

        self.process_noise_std = 0.05
        self.measure_noise_std = 0.02

    def reset(self) -> float:
        self.h = float(np.random.uniform(2.0, 5.0))
        self.step_count = 0
        return self.h

    def step_env(self, u: int) -> float:
        u = int(np.clip(u, phys_min_pump, phys_max_pump))

        self.h = self.a * self.h + self.b * u + self.c + float(np.random.normal(0, self.process_noise_std))
        self.h = float(np.clip(self.h, 0.0, tank_height_cm))

        h_measured = self.h + float(np.random.normal(0, self.measure_noise_std))
        h_measured = float(np.clip(h_measured, 0.0, tank_height_cm))

        self.step_count += 1
        return h_measured


# =============================================================================
# 6. Training
# =============================================================================

def plot_curve(values, title: str):
    plt.figure(figsize=(9, 4))
    plt.plot(values, alpha=0.6)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train():
    agent = DQNAgent()
    env = WaterTankSimulator()

    rewards = []
    losses = []

    for ep in range(num_episodes):
        h = env.reset()
        agent.reset_episode(h)

        ep_reward = 0.0
        ep_losses = []

        for _ in range(max_steps_per_episode):
            state = agent.build_state(h)

            action = agent.select_action(state, training=True)  # 0..10
            u = agent.action_to_u(action)                       # 5..15 (정수)

            h_next = env.step_env(u)

            # 결과 기반 보상 (너가 수정한 방식 유지)
            reward = agent.compute_reward(h_next, u)

            # 내부 상태 업데이트 순서(기존 흐름 유지)
            agent.h_prev = h
            agent.u_prev = float(u)

            agent.error_int = float(np.clip(
                agent.error_int + (setpoint_cm - h) * control_period_s,
                -10.0, 10.0
            ))

            next_state = agent.build_state(h_next)
            agent.buffer.push(state, action, reward, next_state)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            agent.update_target()

            h = h_next
            ep_reward += reward
            agent.steps_done += 1

        agent.decay_epsilon()
        agent.episodes_done += 1

        rewards.append(ep_reward)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (ep + 1) % 10 == 0:
            print(
                f"ep {ep+1}/{num_episodes} | reward {ep_reward:.2f} | "
                f"avg10 {np.mean(rewards[-10:]):.2f} | loss {losses[-1]:.4f} | eps {agent.epsilon:.3f}"
            )

    plot_curve(rewards, "training reward (u=5..15, 11 actions, no time, no done)")
    plot_curve(losses, "training loss (u=5..15, 11 actions, no time, no done)")

    agent.save("dqn_water_level_model.pth")


if __name__ == "__main__":
    train()
