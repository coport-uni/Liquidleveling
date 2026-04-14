"""DQN Training module for Water Tank Liquid Level Control.

This script trains a Deep Q-Network to control a simulated water pump,
aiming to maintain a target liquid level while minimizing abrupt changes
in pump speed (chattering).
"""

import collections
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. System & DQN Configuration

# System Parameters (Exported for validation script)
tank_height_cm = 8.0
setpoint_cm = 4.0
control_period_s = 3.0

# Pump Parameters
min_pump_speed = 0
max_pump_speed = 20
num_actions = max_pump_speed - min_pump_speed + 1

# Dimensions
state_dims = 3   # [h_norm, error_norm, error_int_norm]
action_dims = num_actions

# Hyperparameters
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


# 2. Replay Buffer
Transition = collections.namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state"),
)


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, max_length: int):
        self.buffer = collections.deque(maxlen=max_length)

    def put_data(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ) -> None:
        """Store a new transition in the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state))

    def sample_minibatch(self) -> List[Transition]:
        """Randomly sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# 3. Q-Network (MLP)
class QNetwork(nn.Module):
    """Multilayer perceptron for approximating Q-values."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        hidden_layer1 = 128
        hidden_layer2 = 128

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values for all actions."""
        return self.net(state)


# 4. Agent
class DQNAgent:
    """Deep Q-Network agent for pump speed control."""

    def __init__(self, state_dims: int, action_dims: int):
        self.state_dims = state_dims
        self.action_dims = action_dims

        self.q_net = QNetwork(state_dims, action_dims)
        self.target_net = QNetwork(state_dims, action_dims)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(
            self.q_net.parameters(), lr=learning_rate
        )

        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.steps_done = 0
        self.episodes_done = 0

        # These track control history across steps and feed the reward
        # signal; they are not part of the observation vector.
        self.u_prev = float((min_pump_speed + max_pump_speed) // 2)
        self.error_int = 0.0

        self.training_losses: List[float] = []

    def action_to_u(self, action: int) -> int:
        """Convert an action index back to a physical pump speed."""
        return int(min_pump_speed + action)

    def u_to_action(self, u: int) -> int:
        """Convert a physical pump speed to an action index."""
        return int(u - min_pump_speed)

    def reset_episode(self) -> None:
        """Reset internal tracking variables at the start of an episode."""
        self.u_prev = float((min_pump_speed + max_pump_speed) // 2)
        self.error_int = 0.0

    def build_state(self, h: float) -> np.ndarray:
        """Construct the normalised observation vector.

        Each component is scaled so that the network sees inputs in
        roughly [-1, 1], which keeps gradients well-conditioned.
        """
        error = setpoint_cm - h

        state = np.array([
            h / tank_height_cm,
            error / tank_height_cm,
            np.clip(self.error_int, -10.0, 10.0) / 10.0,
        ], dtype=np.float32)

        return state

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dims - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def compute_reward(self, h: float, u: float) -> float:
        """Return reward R = -|error| - 0.05 * |delta_u|.

        The smoothness term is essential to suppress pump chattering;
        without it the policy oscillates between adjacent speeds.
        """
        error = abs(h - setpoint_cm)
        tracking_penalty = -error

        delta_u = abs(u - self.u_prev)
        smoothness_penalty = -0.05 * delta_u

        return float(tracking_penalty + smoothness_penalty)

    def train_step(self) -> Optional[float]:
        """Perform a single step of mini-batch gradient descent."""
        if len(self.buffer) < min_buffer_size:
            return None

        minibatch = self.buffer.sample_minibatch()

        state_batch = torch.tensor(
            np.array([t.state for t in minibatch]),
            dtype=torch.float32,
        )
        action_batch = torch.tensor(
            [t.action for t in minibatch], dtype=torch.int64
        )
        reward_batch = torch.tensor(
            [t.reward for t in minibatch], dtype=torch.float32
        )
        next_state_batch = torch.tensor(
            np.array([t.next_state for t in minibatch]),
            dtype=torch.float32,
        )

        q_values = (
            self.q_net(state_batch)
            .gather(1, action_batch.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            targets = reward_batch + gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_losses.append(loss_value)

        return loss_value

    def update_target(self) -> None:
        """Soft-update the target network toward the policy network."""
        pairs = zip(
            self.target_net.parameters(), self.q_net.parameters()
        )
        for target_param, policy_param in pairs:
            target_param.data.copy_(
                tau * policy_param.data
                + (1.0 - tau) * target_param.data
            )

    def decay_epsilon(self) -> None:
        """Decay the exploration rate toward its floor."""
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

    def save(self, path: str) -> None:
        """Save the model checkpoint.

        The policy-network state dict is stored under the legacy key
        "qNet" so existing ``.pth`` files remain loadable.
        """
        torch.save({
            "qNet": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
            "steps_done": self.steps_done,
            "training_losses": self.training_losses[-1000:],
        }, path)
        print(f"Model successfully saved to {path}")

    def load(self, path: str) -> None:
        """Load a previously saved model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.q_net.load_state_dict(checkpoint["qNet"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = float(checkpoint["epsilon"])
        self.episodes_done = int(checkpoint.get("episodes_done", 0))
        self.steps_done = int(checkpoint.get("steps_done", 0))

        if "training_losses" in checkpoint:
            self.training_losses = checkpoint["training_losses"]

        print(
            f"Model loaded from {path} | "
            f"Episodes: {self.episodes_done} | "
            f"Epsilon: {self.epsilon:.4f}"
        )


# 5. Environment Simulator
class WaterTankSimulator:
    """First-order difference-equation model of the water tank."""

    def __init__(self):
        # h(k+1) = a*h(k) + b*u(k) + c
        self.a = 0.95
        self.b = 0.05
        self.c = -0.3
        self.h = setpoint_cm

        self.process_noise_std = 0.05
        self.measure_noise_std = 0.02

    def reset(self) -> float:
        """Reset the tank to a random initial liquid level."""
        self.h = float(np.random.uniform(2.0, 5.0))
        return self.h

    def step_env(self, u: int) -> float:
        """Apply pump speed and return the newly measured level."""
        u = int(np.clip(u, min_pump_speed, max_pump_speed))

        process_noise = float(
            np.random.normal(0, self.process_noise_std)
        )
        self.h = (
            self.a * self.h + self.b * u + self.c + process_noise
        )
        self.h = float(np.clip(self.h, 0.0, tank_height_cm))

        measure_noise = float(
            np.random.normal(0, self.measure_noise_std)
        )
        h_measured = self.h + measure_noise
        return float(np.clip(h_measured, 0.0, tank_height_cm))


# 6. Training Routine
def plot_training_results(
    rewards: List[float], losses: List[float]
) -> None:
    """Plot the reward and loss learning curves."""
    plt.figure(figsize=(12, 4))
    plt.suptitle(
        "DQN Water Tank Training Results",
        fontsize=14,
        fontweight="bold",
    )

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.7, color="green")
    plt.title("Episode Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.7, color="red")
    plt.title("Average Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train() -> None:
    """Run the full training loop and save the resulting model."""
    print("=== Starting DQN Training ===")
    print(
        f"State dimensions: {state_dims} "
        f"[h_norm, error_norm, error_int_norm]"
    )
    print(
        f"Action dimensions: {action_dims} "
        f"(speeds {min_pump_speed}~{max_pump_speed})"
    )

    agent = DQNAgent(state_dims, action_dims)
    env = WaterTankSimulator()

    episode_rewards = []
    losses = []

    for ep in range(num_episodes):
        h = env.reset()
        agent.reset_episode()

        ep_reward = 0.0
        ep_losses = []

        for _ in range(max_steps_per_episode):
            state = agent.build_state(h)

            action = agent.select_action(state, training=True)
            u = agent.action_to_u(action)

            h_next = env.step_env(u)

            reward = agent.compute_reward(h_next, u)

            agent.u_prev = float(u)
            agent.error_int = float(np.clip(
                agent.error_int
                + (setpoint_cm - h_next) * control_period_s,
                -10.0,
                10.0,
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

        episode_rewards.append(ep_reward)
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        losses.append(avg_loss)

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1:4d}/{num_episodes} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Loss: {avg_loss:6.4f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("=== Training Completed ===")
    agent.save("dqn_liquid_level_model.pth")
    plot_training_results(episode_rewards, losses)


if __name__ == "__main__":
    train()
