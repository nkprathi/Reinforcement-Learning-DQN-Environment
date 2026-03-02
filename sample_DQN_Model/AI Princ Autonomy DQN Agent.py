# Imports:
# --------
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Continuous Maze Environment:
# -----------------------------
class ContinuousMazeEnv(gym.Env):
    """
    A continuous-state, discrete-action maze environment with danger zones.

    State: 2D continuous position normalized to [0,1]^2
    Actions: 0=up, 1=down, 2=left, 3=right
    Rewards: Define your own reward structure

    Rendering: Pygame window
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
        self.walls = [(0.0, 0.9, 0.4, 1.0),
                      (0.6, 0.9, 1.0, 1.0), 
                      (0.0, 0.0, 0.4, 0.1),
                      (0.6, 0.0, 1.0, 0.1)]

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
        # Create a delta vector matching the dimensions of agent_pos
        delta = np.zeros_like(self.agent_pos, dtype=np.float32)
        # Only modify the first two dimensions (x, y)
        if action == 0:
            delta[1] = self.step_size
        elif action == 1:
            delta[1] = -self.step_size
        elif action == 2:
            delta[0] = -self.step_size
        elif action == 3:
            delta[0] = self.step_size

        new_pos = self.agent_pos + delta
        # Clip to bounds
        new_pos = np.clip(new_pos, 0.0, 1.0)

        # Check collision with walls
        collided = False
        for (xmin, ymin, xmax, ymax) in self.walls:
            if xmin <= new_pos[0] <= xmax and ymin <= new_pos[1] <= ymax:
                collided = True
                break
        
        #! Reward structure:
        """
        - The reward structure can be modified as needed.
        - However, a negative reward should be received for collisions with walls or danger zones.
        """

        if collided:
            reward = -1.0
            # stay in place
            new_pos = self.agent_pos.copy()
        else:
            reward = 0
            self.agent_pos = new_pos

            
        done = False
        # Check danger zones
        for (xmin, ymin, xmax, ymax) in self.danger_zones:
            if xmin <= self.agent_pos[0] <= xmax and ymin <= self.agent_pos[1] <= ymax:
                reward -= 100.0
                done = True
                break

        # Check goal
        if np.linalg.norm(self.agent_pos - self.goal_pos) <= self.goal_radius:
            reward += 10.0
            done = True

        observation = self.agent_pos.copy()
        return observation, reward, done, False, {}

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


# Run as a script: To test the environment
# ----------------
if __name__ == "__main__":
    for _ in range(5):
        env = ContinuousMazeEnv()
        obs, info = env.reset()
        done = False
        for _ in range(100):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            print(f"Obs: {obs}, Reward: {reward}, Done: {done}")

            if done:
                break
    
        env.close()
