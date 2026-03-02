import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim=2, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        
        # Value stream
        self.fc_value = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

        # Advantages stream
        self.fc_advantages = nn.Linear(128, 128)
        self.advantages = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        # Value calc
        v = torch.relu(self.fc_value(x))
        V = self.value(v)

        # Advantages calc
        a = torch.relu(self.fc_advantages(x))
        A = self.advantages(a)

        # Calc Q
        Q = V + A - torch.mean(A, dim=1, keepdim=True)
        return Q


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
