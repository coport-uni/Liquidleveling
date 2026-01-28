import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import matplotlib.pyplot as plt


# 1) 시스템 설정

tank_height_cm = 8.0
setpoint_cm = 4.0
control_period_s = 3.0

# 펌프 범위 (정수)
min_pump_speed = 0
max_pump_speed = 20
num_actions = max_pump_speed - min_pump_speed + 1

# 상태 / 액션 차원
state_dims = 5   # [h, h_prev, error, error_int, u_prev]
action_dims = num_actions

# DQN 파라미터
learning_rate = 1e-4
gamma = 0.99
tau = 0.005

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995

buffer_size = 10000
batch_size = 64
min_buffer_size = 1000

max_steps_per_episode = 100
num_episodes = 1000


# 2) Replay Buffer

Transition = collections.namedtuple("Transition", ("state", "action", "reward", "next_state"))

class ReplayBuffer:
    def __init__(self, max_length):
        self.buffer = collections.deque(maxlen=max_length)

    def put_data(self, *args):
        # args = (state, action, reward, next_state)
        self.buffer.append(Transition(*args))

    def sample_minibatch(self):

        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 3) Q network (MLP)

class Qnetwork(nn.Module):
    def __init__(self, state_dims:int, action_dims:int):
        super().__init__()

        hidden_layer1 = 128
        hidden_layer2 = 128

        self.net = nn.Sequential(
            nn.Linear(state_dims, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, action_dims)
        )

    def forward(self, state:torch.Tensor):
        q_value = self.net(state)
        return q_value


# 4) Agent

class DQNAgent:
    def __init__(self, state_dims:int, action_dims:int):
        # Network
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.qNet = Qnetwork(state_dims, action_dims)
        self.target_net = Qnetwork(state_dims, action_dims)
        self.target_net.load_state_dict(self.qNet.state_dict())
        self.optimizer = optim.AdamW(self.qNet.parameters(), lr=learning_rate)
        
        self.buffer = ReplayBuffer(buffer_size)

        # Exploration
        self.epsilon = epsilon_start
        self.steps_done = 0
        self.episodes_done = 0

        # State tracking
        self.h_prev = setpoint_cm
        self.u_prev = float((min_pump_speed + max_pump_speed) // 2)
        self.error_int = 0.0
        
        self.training_losses = []
        self.episode_rewards = []

    def action_to_u(self, action: int) -> int:
        """Action index to pump speed"""
        return int(min_pump_speed + action)

    def u_to_action(self, u: int) -> int:
        """Pump speed to action index"""
        return int(u - min_pump_speed)

    def reset_episode(self, h0: float):
        self.h_prev = float(h0)
        self.u_prev = float((min_pump_speed + max_pump_speed) // 2)
        self.error_int = 0.0

    def build_state(self, h: float) -> np.ndarray:
        """현재 관측을 state vector로 변환 (정규화 포함)"""
        error = setpoint_cm - h
        
        state = np.array([
            h / tank_height_cm,                          # [0, 1]
            self.h_prev / tank_height_cm,                # [0, 1]
            error / tank_height_cm,                      # [-1, 1]
            np.clip(self.error_int, -10.0, 10.0) / 10.0, # [-1, 1]
            self.u_prev / max_pump_speed,                # [0, 1]
        ], dtype=np.float32)
        
        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        # Exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, action_dims - 1)

        # Exploitation
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.qNet(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def compute_reward(self, h: float, u: float, done: bool = False) -> float:
        # 1) Tracking error penalty (quadratic)
        error = abs(h - setpoint_cm)
        if error < 0.20:
            tracking = 20.0 * (1.0 - error/0.15)
        else:
            tracking = -15.0 * (error ** 2)

        # 2) Control smoothness penalty
        du = abs(u - self.u_prev)
        smooth = -0.1 * (du ** 2)

        return float(tracking + smooth)

    def train_step(self):
        """Single training step with minibatch"""
        if len(self.buffer) < min_buffer_size:
            return None

        # Sample minibatch
        minibatch = self.buffer.sample_minibatch()

        # Convert to tensors
        state_batch = torch.tensor(np.array([t.state for t in minibatch]), dtype=torch.float32)
        action_batch = torch.tensor([t.action for t in minibatch], dtype=torch.int64)
        reward_batch = torch.tensor([t.reward for t in minibatch], dtype=torch.float32)
        next_state_batch = torch.tensor(np.array([t.next_state for t in minibatch]), dtype=torch.float32)
        
        # Current Q-values: Q(s, a)
        q_values = self.qNet(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            targets = reward_batch + gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qNet.parameters(), 1.0)
        
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        
        return loss_value

    def update_target(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.qNet.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )

    def decay_epsilon(self):
        """Epsilon decay with minimum bound"""
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

    def get_training_stats(self):
        """학습 통계 반환"""
        if not self.training_losses:
            return {}
        
        return {
            'avg_loss': np.mean(self.training_losses[-100:]),
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer),
            'episodes_done': self.episodes_done,
            'steps_done': self.steps_done
        }

    def save(self, path: str):
        """Model checkpoint 저장"""
        torch.save({
            "qNet": self.qNet.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
            "steps_done": self.steps_done,
            "training_losses": self.training_losses[-1000:],
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Model checkpoint 로드"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.qNet.load_state_dict(checkpoint["qNet"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint["epsilon"])
        self.episodes_done = int(checkpoint.get("episodes_done", 0))
        self.steps_done = int(checkpoint.get("steps_done", 0))
        
        if "training_losses" in checkpoint:
            self.training_losses = checkpoint["training_losses"]
        
        print(f"Model loaded from {path}")
        print(f"  - Episodes: {self.episodes_done}")
        print(f"  - Epsilon: {self.epsilon:.4f}")


# 5. Simulator

class WaterTankSimulator:
    def __init__(self):
        self.a = 0.95
        self.b = 0.05
        self.c = -0.3
        self.h = setpoint_cm
        self.step_count = 0

        self.process_noise_std = 0.05
        self.measure_noise_std = 0.02

    def reset(self) -> float:
        self.h = float(np.random.uniform(2.0, 5.0))
        self.step_count = 0
        return self.h

    def step_env(self, u: int) -> float:
        u = int(np.clip(u, min_pump_speed, max_pump_speed))

        self.h = self.a * self.h + self.b * u + self.c + float(np.random.normal(0, self.process_noise_std))
        self.h = float(np.clip(self.h, 0.0, tank_height_cm))

        h_measured = self.h + float(np.random.normal(0, self.measure_noise_std))
        h_measured = float(np.clip(h_measured, 0.0, tank_height_cm))

        self.step_count += 1
        return h_measured


# 6) Training

def plot_curve(values, title: str):
    plt.figure(figsize=(9, 4))
    plt.plot(values, alpha=0.6)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_results(rewards, losses):
    plt.figure(figsize=(12, 4))

    plt.suptitle("DQN Water Tank Training Results", fontsize=14, fontweight="bold")

    # Reward plot
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.7)
    plt.title("Episode Reward")
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.7)
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train():
    print("==학습시작==")
    agent = DQNAgent(state_dims, action_dims)
    env = WaterTankSimulator()

    reward_batch = []
    losses = []

    for ep in range(num_episodes):
        h = env.reset()
        agent.reset_episode(h)

        ep_reward = 0.0
        ep_losses = []

        for _ in range(max_steps_per_episode):
            state = agent.build_state(h)

            action = agent.select_action(state, training=True)
            u = agent.action_to_u(action)

            h_next = env.step_env(u)

            # 결과 기반 보상
            reward = agent.compute_reward(h_next, u)

            # 내부 상태 업데이트 순서
            agent.h_prev = h
            agent.u_prev = float(u)

            agent.error_int = float(np.clip(
                agent.error_int + (setpoint_cm - h) * control_period_s,
                -10.0, 10.0
            ))

            next_state = agent.build_state(h_next)
            agent.buffer.put_data(state, action, reward, next_state)

            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            agent.update_target()

            h = h_next
            ep_reward += reward
            agent.steps_done += 1

        agent.decay_epsilon()
        agent.episodes_done += 1

        reward_batch.append(ep_reward)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (ep + 1) % 10 == 0:
            print(
                f"ep {ep+1}/{num_episodes} | reward {ep_reward:.2f} | "
                f"avg10 {np.mean(reward_batch[-10:]):.2f} | loss {losses[-1]:.4f} | eps {agent.epsilon:.3f}"
            )

    plot_training_results(reward_batch, losses)

    agent.save("dqn_water_level_model.pth")


if __name__ == "__main__":
    train()
