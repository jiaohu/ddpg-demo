import random

import numpy as np
import torch
from torch import optim, nn

from actor import Actor, Critic


class DDPG:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=1e-2, buffer_size=10000, batch_size=128):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # 初始化目标网络的参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 定义经验回放缓冲区
        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(buffer_size)

        # 其他超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()
        return action.squeeze()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample(self.batch_size)

        # 计算 target Q 值
        next_action = self.actor_target(next_state_batch)
        target_value = self.critic_target(next_state_batch, next_action.detach())
        expected_value = reward_batch + self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        # 计算 critic loss
        value = self.critic(state_batch, action_batch)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        # 更新 critic 网络
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # 计算 actor loss
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # 更新 actor 网络
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state)
        )

    def __len__(self):
        return len(self.memory)
