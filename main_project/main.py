import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from collections import deque
import random

# Import local modules
from env import ContinuousMazeEnv
from DQN_model import DQN, ReplayBuffer
from utils import DualLogger, select_action

import matplotlib.pyplot as plt
import os
from datetime import datetime

if __name__ == "__main__":

    # -----------------------------
    # HYPERPARAMETERS
    # -----------------------------
    LEARNING_RATE = 1e-3
    REPLAY_CAPACITY = 50000
    BATCH_SIZE = 64
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 100
    MAX_STEPS_PER_EPISODE = 300
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.995
    # -----------------------------
    # 1. Setup Experiment Folder
    # --------------------------
    base_result_dir = "results"
    os.makedirs(base_result_dir, exist_ok=True)

    user_input = input("Enter experiment name (default: timestamp): ").strip()
    if not user_input:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    else:
        exp_name = user_input
    
    experiment_dir = os.path.join(base_result_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Saving results to: {experiment_dir}")

    # Initialize logging with path inside experiment_dir
    log_path = os.path.join(experiment_dir, "dqn_training.log")
    sys.stdout = DualLogger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Window visible during training; faster FPS for quicker training but still visible
    env = ContinuousMazeEnv(render_mode="human")
    env.metadata["render_fps"] = 60  # Speed up rendering a bit

    dqn = DQN().to(device)
    target_dqn = DQN().to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    obs, info = env.reset()
    
    # Training Parameters
    epsilon = EPSILON_START
    epsilon_min = EPSILON_MIN 
    epsilon_decay = EPSILON_DECAY 
    
    success_streak = 0
    episode_reward = 0
    episode = 0
    steps_done = 0
    episode_steps = 0 # Track steps in current episode
    
    # Tracking for plots
    rewards_history = []
    
    print("Training DQN - Running untill 100 consecutive successes with low epsilon...")

    while success_streak < 100:
        env.render()
        if env.screen is None:  # Window closed manually
            break

        # Action Selection
        action = select_action(dqn, obs, epsilon, env.action_space, device)

        next_obs, reward, terminated, truncated= env.step(action)
        
        # Check episode step limit
        episode_steps += 1
        if episode_steps >= MAX_STEPS_PER_EPISODE:
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
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
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
                
                targets = rewards_t + GAMMA * (1 - dones_t) * next_q_values
            
            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update Target Network
        if steps_done % TARGET_UPDATE_FREQ == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        # Episode Complete
        if done:
            episode += 1
            rewards_history.append(episode_reward)
            
            # Check Success (Distance Based)
            dist_to_goal = np.linalg.norm(obs - env.goal_pos)
            is_success = dist_to_goal <= env.goal_radius
            
            # Update Epsilon
            if epsilon > epsilon_min:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Update Streak
            if is_success:
                # Logic: We only count streak if we are AT the minimum epsilon
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
        
        # Save model
        model_path = os.path.join(experiment_dir, "dqn_model.pth")
        torch.save(dqn.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_history, label='Episode Reward', alpha=0.5)
        
        # Calculate Simple Moving Average (SMA)
        window_size = 50
        if len(rewards_history) >= window_size:
            sma = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
            # x-axis adjustment for "valid" convolution (starts at window_size-1)
            plt.plot(range(window_size-1, len(rewards_history)), sma, label=f'SMA {window_size}', color='orange', linewidth=2)
        
        plt.title("DQN Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plot_path = os.path.join(experiment_dir, "training_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
