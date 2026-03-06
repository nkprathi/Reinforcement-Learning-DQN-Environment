# Custom Continuous Maze Solver using Dueling Double DQN (D3QN)

## 1. Project Overview & Environment

**The Goal:** Train an RL agent to navigate a custom 2D maze with continuous state space, avoiding danger zones and walls, to reach a specified goal using a Deep Q-Network. The training stops when the agent successfully reaches the goal for 100 consecutive episodes.

**The Environment:** 
*   **Observation Space (`Continuous`):** The state is a 2D continuous space `[x, y]` representing the agent's normalized position in the grid `[0, 1]^2`.
*   **Action Space (`Discrete`):** 4 discrete actions (0: Up, 1: Down, 2: Left, 3: Right), moving the agent by a continuous step size of 0.05.

**The Frameworks:**
*   **PyTorch:** For defining and training the neural network.
*   **Gymnasium:** Custom environment design (`ContinuousMazeEnv`).
*   **Pygame:** For rendering the environment for human visualization.
*   **Matplotlib:** For plotting the reward curves and moving averages.

---

## 2. Core Architecture Notes

The agent leverages a **Dueling Double Deep Q-Network (D3QN)**, which separates the estimation of the state value and the action advantages, leading to better and more stable convergence.

*   **Network Topology:** 
    *   Shared Layer: 1 Fully Connected layer with 128 neurons.
    *   Value Stream: 1 Hidden layer (128 neurons) $\rightarrow$ Output (1 neuron representing State Value).
    *   Advantage Stream: 1 Hidden layer (128 neurons) $\rightarrow$ Output (4 neurons representing Action Advantages).
*   **DQN Variants Used:**
    *   **Dueling Architecture:** Aggregates streams using $Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')$.
    *   **Double DQN ($\text{DDQN}$):** Reduces overestimation bias by using the *online* network to select the best action and the *target* network to estimate its Q-value during loss calculation.
*   **Activation Functions:** $\text{ReLU}$ for all hidden layers.

---

## 3. Hyperparameter Log

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | $1 \times 10^{-3}$ | Step size for the Adam optimizer. |
| **Discount Factor ($\gamma$)** | $0.99$ | Importance of future rewards over immediate ones. |
| **Replay Buffer Size** | $50,000$ | Number of past transitions $(s, a, r, s', d)$ stored for experience replay. |
| **Batch Size** | $64$ | Number of samples drawn from the buffer per training step. |
| **Epsilon ($\epsilon$) Start**| $1.0$ | Initial exploration rate. |
| **Epsilon ($\epsilon$) Decay**| $0.995$ | Multiplicative decay rate applied to epsilon when a goal is reached. |
| **Epsilon ($\epsilon$) Min**  | $0.1$ | Minimum exploration rate ensuring continual random exploration. |
| **Target Update Freq** | $100$ | Steps indicating how often the target network weights are updated. |
| **Max Steps / Episode**| $300$ | Maximum steps allowed in the environment to prevent infinite loops. |

---

## 4. Training Performance & Results

*   **Reward Curves:** The agent's performance in terms of overall reward per episode is plotted automatically at the end of training.
*   **Moving Average:** A 50-episode Simple Moving Average (SMA 50) is included in the plot to smooth out the inherent noise from exploration.
*   **Success Metric:** Training terminates when **100 consecutive successes** are logged while the agent operates at the minimum epsilon threshold ($\epsilon \le 0.1$). Ensure you check the `results/` directory for the `training_plot.png`.

---

## 5. Challenges & Lessons Learned

*   **Reward Engineering & Local Optima:** Initially, distance-to-goal was considered for reward shaping. However, giving dense rewards based strictly on distance often causes the agent to get stuck behind walls (local optima). Hence, distance shaping was disabled in favor of:
    *   Step penalty: `-0.05`
    *   Wall collision: `-1.0`
    *   Target completion: `+100.0`
    *   Danger zone: `-100.0`
    *   Time limit exceeded: `-50.0`
*   **Stability Enhancements:** Implementing both *Dueling DQN* and *Double DQN* significantly reduced the Q-value overestimation seen in classic DQNs, leading to stabler action evaluations around the danger zones.

---

## 6. How to Run

**1. Installation:**
Ensure you have `uv` or `pip` installed, then set up your environment:
```bash
pip install torch numpy gymnasium pygame matplotlib
```

**2. Training:**
To train the model from scratch (you will be prompted to enter an experiment name, leaving it blank defaults to a timestamp):
```bash
python main.py
```

**3. Artifacts & Outputs:**
Check the `results/` folder for:
*   `dqn_model.pth`: The final trained model weights.
*   `dqn_training.log`: The training logs.
*   `training_plot.png`: Performance curve visualization.
