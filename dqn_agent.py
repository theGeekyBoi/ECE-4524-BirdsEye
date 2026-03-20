"""
DQN agent: network architecture, replay buffer, and training logic.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from car_env import STATE_SIZE, NUM_ACTIONS


class DQNetwork(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size=STATE_SIZE,
        action_size=NUM_ACTIONS,
        lr=5e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.997,
        buffer_capacity=200_000,
        batch_size=128,
        tau=0.005,
        device=None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.hard_update_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net picks the best action, target net evaluates it
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.soft_update_target()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def soft_update_target(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def hard_update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.policy_net.eval()
