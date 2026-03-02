# Imports:
# --------
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys

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


# Continuous Maze Environment:
# -----------------------------
class ContinuousMazeEnv(gym.Env):
    """
    A continuous-state, discrete-action maze environment with danger zones.

    State: 2D continuous position normalized to [0,1]^2
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.width = 600
        self.height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Discrete actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Continuous observations: x, y in [0,1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        #! Don't modify the step size
        self.step_size = 0.05  # normalized units per step

        self.agent_pos = None

        #! NOTE: The goal, danger zone and wall positions and dimensions should NOT be altered

        # Goal region
        self.goal_pos = np.array([0.9, 0.5], dtype=np.float32)
        self.goal_radius = 0.05

        # Danger zones: list of normalized rectangles (xmin, ymin, xmax, ymax)
        self.danger_zones = [(0.4, 0.3, 0.6, 0.7)]

        # Maze walls: list of normalized rectangles (xmin, ymin, xmax, ymax)
        self.walls = [
            (0.0, 0.9, 0.4, 1.0),
            (0.6, 0.9, 1.0, 1.0),
            (0.0, 0.0, 0.4, 0.1),
            (0.6, 0.0, 1.0, 0.1),
        ]

    # Method 1:
    # ---------
    def reset(self, *, seed=None, options=None):  # gymnasium signature
        super().reset(seed=seed)
        # Start near bottom-left
        self.agent_pos = np.array([0.1, 0.5], dtype=np.float32)
        observation = self.agent_pos.copy()
        info = {}
        return observation, info

    # Method 2:
    # ---------
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        
        # Calculate distance to goal BEFORE moving
        dist_old = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Create a delta vector matching the dimensions of agent_pos
        delta = np.zeros_like(self.agent_pos, dtype=np.float32)

        # Only modify the first two dimensions (x, y)
        if action == 0:      # up
            delta[1] = self.step_size
        elif action == 1:    # down
            delta[1] = -self.step_size
        elif action == 2:    # left
            delta[0] = -self.step_size
        elif action == 3:    # right
            delta[0] = self.step_size

        new_pos = self.agent_pos + delta

        # Clip to bounds
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        # Calculate distance to goal AFTER moving (before collision reset)
        # This rewards "trying" to move closer, even if blocked, though wall penalty overrides.
        # Typically shaping is based on state_t and state_{t+1}, so we use the actual resulting position.
        
        # Check collision with walls
        collided = False
        for (xmin, ymin, xmax, ymax) in self.walls:
            if xmin <= new_pos[0] <= xmax and ymin <= new_pos[1] <= ymax:
                collided = True
                break
        
        # Determine actual next state
        if collided:
             final_pos = self.agent_pos.copy() # Stay in place
        else:
             final_pos = new_pos
        
        dist_new = np.linalg.norm(final_pos - self.goal_pos)
        
        # REWARD SHAPING
        # 1. Step cost
        reward = -0.05
        
        reward += 0.0 # Disabled distance shaping to prevent local optima
        
        if collided:
            reward = -1.0
            self.agent_pos = final_pos
        else:
            self.agent_pos = final_pos
            
        terminated = False

        # Check danger zones
        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            if xmin <= self.agent_pos[0] <= xmax and ymin <= self.agent_pos[1] <= ymax:
                reward = -100.0 # Danger penalty
                terminated = True
                break

        # Goal (make this *large* compared to everything else)
        if np.linalg.norm(self.agent_pos - self.goal_pos) <= self.goal_radius:
            reward = 100.0
            terminated = True

        observation = self.agent_pos.copy()
        return observation, reward, terminated, False, {}

    # Method 3:
    # ---------
    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Continuous Maze Environment")
            self.clock = pygame.time.Clock()

        # Process window events so it stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return # Stop rendering if quit

        if self.screen is None: # Double check in case close() was called
            return

        # Draw background
        self.screen.fill((255, 255, 255))

        # Draw walls (black)
        for (xmin, ymin, xmax, ymax) in self.walls:
            rect = pygame.Rect(
                xmin * self.width,
                self.height - ymax * self.height,
                (xmax - xmin) * self.width,
                (ymax - ymin) * self.height,
            )
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # Draw danger zones (red)
        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            rect = pygame.Rect(
                xmin * self.width,
                self.height - ymax * self.height,
                (xmax - xmin) * self.width,
                (ymax - ymin) * self.height,
            )
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw goal (green circle)
        goal_pix = (
            int(self.goal_pos[0] * self.width),
            int(self.height - self.goal_pos[1] * self.height),
        )
        pygame.draw.circle(self.screen, (0, 255, 0), goal_pix, int(self.goal_radius * self.width))

        # Draw agent (blue circle)
        agent_pix = (
            int(self.agent_pos[0] * self.width),
            int(self.height - self.agent_pos[1] * self.height),
        )
        pygame.draw.circle(self.screen, (0, 0, 255), agent_pix, 10)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    # Method 4:
    # ---------
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# --------------------
# DQN components
# --------------------

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


def select_action(dqn, state, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn(state_t)
    return int(torch.argmax(q_values, dim=1).item())




# Run as a script: To train and then show environment with text output
# ----------------
if __name__ == "__main__":
    # Initialize logging
    sys.stdout = DualLogger("dqn_training.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Window visible during training; faster FPS for quicker training but still visible
    env = ContinuousMazeEnv(render_mode="human")
    env.metadata["render_fps"] = 60  # Speed up rendering a bit

    dqn = DQN().to(device)
    target_dqn = DQN().to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(50000)

    obs, info = env.reset()
    
    # Training Parameters
    epsilon = 1.0
    epsilon_min = 0.1 #0.05 # Floor for epsilon
    epsilon_decay = 0.995 # Decay per episode (much faster for demo)
    
    success_streak = 0
    episode_reward = 0
    episode = 0
    steps_done = 0
    episode_steps = 0 # Track steps in current episode
    
    print("Training DQN - Running untill 100 consecutive successes with low epsilon...")

    while success_streak < 100:
        env.render()
        if env.screen is None:  # Window closed manually
            break

        # Action Selection
        action = select_action(dqn, obs, epsilon, env.action_space, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Check episode step limit
        episode_steps += 1
        if episode_steps >= 300:
             truncated = True
             reward -= 50.0 # Time limit penalty
        
        done = terminated or truncated

        # Store transition
        replay_buffer.push(obs, action, reward, next_obs, done)

        # Update stats
        episode_reward += reward
        steps_done += 1
        obs = next_obs
        
        # Train Network
        if len(replay_buffer) >= 64:
            states, actions, rewards, next_states, dones = replay_buffer.sample(64)
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

            q_values = dqn(states_t).gather(1, actions_t)
            with torch.no_grad():
                # Double DQN Logic:
                # 1. Select best action using ONLINE network
                best_actions_from_policy = dqn(next_states_t).argmax(dim=1)
                
                # 2. Evaluate that action using TARGET network
                next_q_values = target_dqn(next_states_t).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1))
                
                targets = rewards_t + 0.99 * (1 - dones_t) * next_q_values
            
            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update Target Network
        if steps_done % 100 == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        # Episode Complete
        if done:
            episode += 1
            
            # Check Success (Distance Based)
            dist_to_goal = np.linalg.norm(obs - env.goal_pos)
            is_success = dist_to_goal <= env.goal_radius
            
            # Update Epsilon
            if epsilon > epsilon_min:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Update Streak
            if is_success:
                if epsilon <= epsilon_min + 1e-6:
                     success_streak += 1
                else:
                    success_streak = 0
            else:
                success_streak = 0

            print(f"Episode {episode}: Reward={episode_reward:.1f}, Steps={episode_steps}, Eps={epsilon:.3f}, Streak={success_streak}, Success={is_success}")
            
            episode_reward = 0
            episode_steps = 0
            obs, info = env.reset()

    env.close()
    if success_streak >= 100:
        print("Training Complete: 100 consecutive successes achieved!")

