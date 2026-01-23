"""
=============================================================================
수위 제어 시스템 - DQN 강화학습
- Action = 절대 펌프값 u ∈ {5..15} (정수), 총 11개 액션
- 상태에서 time 제거 버전
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
# 1. 시스템 설정
# =============================================================================

TANK_HEIGHT_CM = 8.0
SETPOINT_CM = 3.85
CONTROL_PERIOD_S = 3.0

# 물리적으로 가능한 펌프 범위(참고용)
PHYS_MIN_PUMP = 0
PHYS_MAX_PUMP = 20

# 학습/제어에서 사용할 펌프 범위 (정수)
ACT_U_MIN = 5
ACT_U_MAX = 15
NUM_ACTIONS = ACT_U_MAX - ACT_U_MIN + 1  # 11

# ---- 상태 / 액션 ----
STATE_DIM = 5   # [h, h_prev, error, error_int, u_prev]
ACTION_DIM = NUM_ACTIONS  # 11

# ---- DQN 파라미터 ----
LEARNING_RATE = 1e-4
GAMMA = 0.99
TAU = 0.005

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

BUFFER_SIZE = 100000
BATCH_SIZE = 64
MIN_BUFFER_SIZE = 1000

MAX_STEPS_PER_EPISODE = 100
NUM_EPISODES = 1000

SAFETY_H_MIN = 1.0
SAFETY_H_MAX = 7.0


# =============================================================================
# 2. Replay Buffer
# =============================================================================

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return (
            torch.tensor(np.array(batch.state), dtype=torch.float32),
            torch.tensor(batch.action, dtype=torch.int64),
            torch.tensor(batch.reward, dtype=torch.float32),
            torch.tensor(np.array(batch.next_state), dtype=torch.float32),
            torch.tensor(batch.done, dtype=torch.float32),  # 경고 방지 (bool -> float)
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 3. DQN Network
# =============================================================================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4. DQN Agent (time 제거, 11개 action: u=5..15)
# =============================================================================

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.episodes_done = 0

        self.h_prev = SETPOINT_CM
        self.u_prev = float((ACT_U_MIN + ACT_U_MAX) // 2)  # 10
        self.error_int = 0.0

    # action(0..10) -> u(5..15)
    def action_to_u(self, action: int) -> int:
        return int(ACT_U_MIN + action)

    def u_to_action(self, u: int) -> int:
        # (디버깅/기록용) u(5..15)->action(0..10)
        return int(u - ACT_U_MIN)

    def reset_episode(self, h0):
        self.h_prev = float(h0)
        self.u_prev = float((ACT_U_MIN + ACT_U_MAX) // 2)
        self.error_int = 0.0

    def build_state(self, h: float) -> np.ndarray:
        error = SETPOINT_CM - h
        # u_prev는 실제 u(5..15)라서 정규화는 물리 최대(20) 또는 ACT_U_MAX(15) 중 택일 가능
        # 여기선 물리 최대(20) 기준 유지 (기존과 일관)
        return np.array([
            h / TANK_HEIGHT_CM,
            self.h_prev / TANK_HEIGHT_CM,
            error / TANK_HEIGHT_CM,
            np.clip(self.error_int, -10.0, 10.0) / 10.0,
            self.u_prev / PHYS_MAX_PUMP,
        ], dtype=np.float32)

    def select_action(self, state, training=True) -> int:
        # ε-greedy
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        with torch.no_grad():
            q = self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))[0]
            return int(torch.argmax(q).item())

    def compute_reward(self, h: float, u: float) -> float:
        # 1) tracking penalty
        error = abs(h - SETPOINT_CM)
        tracking = 0.0 if error < 0.05 else -10.0 * (error ** 2)

        # 2) smoothness penalty (절대 액션에서도 중요)
        du = abs(u - self.u_prev)
        smooth = -0.1 * (du ** 2)

        # 3) safety penalty
        safety = 0.0
        if h > SAFETY_H_MAX:
            safety = -100.0
        elif h < SAFETY_H_MIN:
            safety = -50.0

        # 4) bonus
        bonus = 5.0 if error < 0.1 else 0.0

        return float(tracking + smooth + safety + bonus)

    def train_step(self):
        if len(self.buffer) < MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            target = rewards + (1.0 - dones) * GAMMA * q_next

        loss = nn.functional.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        for t, p in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t.data.copy_(TAU * p.data + (1 - TAU) * t.data)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt["epsilon"])
        self.episodes_done = int(ckpt.get("episodes_done", 0))
        print(f"Model loaded from {path}")


# =============================================================================
# 5. Simulator
# =============================================================================

class WaterTankSimulator:
    def __init__(self):
        self.a, self.b, self.c = 0.95, 0.05, -0.1
        self.h = SETPOINT_CM
        self.step_count = 0

        self.process_noise_std = 0.05
        self.measure_noise_std = 0.02

    def reset(self):
        self.h = float(np.random.uniform(2.0, 5.0))
        self.step_count = 0
        return self.h

    def step_env(self, u: int):
        u = int(np.clip(u, PHYS_MIN_PUMP, PHYS_MAX_PUMP))

        self.h = self.a * self.h + self.b * u + self.c + float(np.random.normal(0, self.process_noise_std))
        self.h = float(np.clip(self.h, 0.0, TANK_HEIGHT_CM))

        h_measured = self.h + float(np.random.normal(0, self.measure_noise_std))
        h_measured = float(np.clip(h_measured, 0.0, TANK_HEIGHT_CM))

        self.step_count += 1
        done = (
            self.step_count >= MAX_STEPS_PER_EPISODE or
            self.h > SAFETY_H_MAX or
            self.h < SAFETY_H_MIN
        )
        return h_measured, done


# =============================================================================
# 6. Training
# =============================================================================

def plot_curve(values, title):
    plt.figure(figsize=(9, 4))
    plt.plot(values, alpha=0.6)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train():
    agent = DQNAgent()
    env = WaterTankSimulator()

    rewards, losses = [], []

    for ep in range(NUM_EPISODES):
        h = env.reset()
        agent.reset_episode(h)

        ep_reward = 0.0
        ep_losses = []

        for _ in range(MAX_STEPS_PER_EPISODE):
            state = agent.build_state(h)

            action = agent.select_action(state, training=True)   # 0..10
            u = agent.action_to_u(action)                        # 5..15 (정수)

            h_next, done = env.step_env(u)

            # reward는 현재 h 기준(기존 유지). 결과 기반으로 바꾸고 싶으면 h_next로 계산해도 됨.
            reward = agent.compute_reward(h, u)

            # 적분 업데이트(스텝당 1회)
            agent.error_int = float(np.clip(
                agent.error_int + (SETPOINT_CM - h) * CONTROL_PERIOD_S,
                -10.0, 10.0
            ))

            next_state = agent.build_state(h_next)
            agent.buffer.push(state, action, reward, next_state, float(done))

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            agent.update_target()

            # 다음 스텝 준비
            agent.h_prev = h
            agent.u_prev = float(u)
            h = h_next
            ep_reward += reward
            agent.steps_done += 1

            if done:
                break

        agent.decay_epsilon()
        agent.episodes_done += 1

        rewards.append(ep_reward)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1}/{NUM_EPISODES} | Reward {ep_reward:.2f} | "
                  f"Avg10 {np.mean(rewards[-10:]):.2f} | Loss {losses[-1]:.4f} | ε {agent.epsilon:.3f}")

    plot_curve(rewards, "Training Reward (u=5..15, 11 actions, no time)")
    plot_curve(losses, "Training Loss (u=5..15, 11 actions, no time)")

    agent.save("dqn_water_level_model_u11_notime.pth")


if __name__ == "__main__":
    train()
