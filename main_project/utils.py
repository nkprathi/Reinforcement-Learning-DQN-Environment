import sys
import torch
import random
import numpy as np

# Logger to write to both console and file
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def select_action(dqn, state, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn(state_t)
    return int(torch.argmax(q_values, dim=1).item())
