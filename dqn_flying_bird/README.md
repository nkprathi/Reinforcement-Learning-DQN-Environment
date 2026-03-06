# Deep Q-Learning: Flappy Bird Solver

## 1. The Environment Specification

The project uses the `FlappyBird-v0` environment from the `flappy_bird_gymnasium` library. 

*   **State Space:** The `use_lidar=False` configuration implies the agent interprets the environment directly from an internal state array representation (not raw pixel inputs). This array typically represents spatial coordinates like the bird's vertical position, velocity, and the horizontal/vertical distances to the next set of pipes.
*   **Action Space:** **Discrete(2)**. The agent has two explicit choices:
    *   `0`: Do nothing (Glide down due to gravity)
    *   `1`: Flap (Apply upward velocity)
*   **Reward Function:** The goal is to maximize the score by passing pipes and staying alive. The standard Flappy Bird reward structure typically gives a small positive reward (e.g., `+0.1`) for every frame the bird stays alive, a large reward (`+1.0`) for passing a pipe, and a harsh penalty (e.g., `-1.0`) for crashing into a pipe or the ground.

## 2. Preprocessing & Input Pipeline

Unlike traditional vision-based DQN models that train directly from pixel data (requiring grayscaling and frame-stacking), our `flappybird1` implementation leverages the internal `flappy_bird_gymnasium` features.
By disabling LIDAR (`use_lidar: False`), the agent directly maps the physics-based environment array (e.g., bird $y$-axis, vertical velocity, distance to upper/lower pipes) into state tensors (`torch.tensor(state, dtype=torch.float)`). Because physics values (such as velocity) inherently capture movement context, explicit temporal frame-stacking is not mandatory for convergence here.

## 3. The "Flying Bird" Architecture

Since we operate on numeric feature arrays rather than pixel images, we bypass Convolutional Neural Networks (CNNs) in favor of deep Fully Connected (Dense) structures utilizing the **Dueling Double Deep Q-Network (D3QN)** architecture.

*   **Fully Connected Layers:**
    *   The `flappybird1` config utilizes a robust hidden layer with **512 nodes** (`fc1_nodes: 512`) followed by a ReLU activation to parse the incoming environment array.
*   **Dueling DQN:** Instead of outputting the final Q-values directly from the hidden layer, the architecture splits into two separate streams:
    *   **Value Stream:** Calculates $V(s)$, depicting how 'good' the state is independent of the action.
    *   **Advantage Stream:** Calculates $A(s, a)$, capturing the specific advantage of Flapping vs. Gliding in that particular state.
    *   These streams are aggregated to produce the final Q-values.
*   **Target Network (Double DQN):** We use a primary *Policy Network* to select actions and a separate, slowly-updating *Target Network* to compute the expected Q-value targets. This decoupling stabilizes learning and prevents the "moving target" destabilization common in vanilla DQNs. Setting `"enable_double_dqn: True"` means we use the policy network to select the *best action* inside the target calculation, thereby reducing overestimation bias.

## 4. Hyperparameter Calibration

Flappy Bird presents a highly volatile environment where a single badly-timed flap guarantees a delayed crash. Carefully tuned hyperparameters are crucial. Our settings (`flappybird1` from `hyperparameters.yml`):

| Parameter | Value | Reasoning |
| :--- | :--- | :--- |
| **Discount Factor ($\gamma$)** | $0.99$ | A high value prioritizes long-term survival (passing future pipes) over immediate, short-sighted rewards. |
| **Exploration ($\epsilon$) Init** | $1.0$ | The agent starts by taking $100\%$ random actions to explore the physics of gravity and flapping. |
| **Exploration ($\epsilon$) Decay**| $0.99995$| Crucially slow decay. Flappy Bird requires a *lot* of trial and error to reliably find the small gap between pipes. |
| **Exploration ($\epsilon$) Min** | $0.05$ | Ensures the agent maintains a $5\%$ exploration rate indefinitely to prevent getting stuck in local optima. |
| **Replay Memory** | $100,000$ | Large enough to remember rare successful pipe passes alongside the frequent crashes during early training. |
| **Network Sync Rate** | $10$ steps | The Target Network weights are frequently updated with the Policy Network weights to keep learning stable but relevant. |

## 5. Performance Metrics

During training, performance is logged actively and graphed automatically via Matplotlib:
*   **Mean Rewards Curve:** Visualizes the agent progressing from continuous early crashes (negative/low scores) to finding the pipeline gap, charting the `Mean Reward` over a rolling 100-episode window.
*   **Epsilon Decay Plot:** Allows for visualizing the exploration-exploitation handoff dynamically alongside the agent's reward growth.
*   *Note: Training artifacts (`.log`, `.pt` model file, and `.png` graphs) are saved directly in the `runs/` directory.*

## 6. Challenges & Solutions

*   **The "Flap Happy" Problem:** In early training phases, random exploration inevitably leads the bird to flap uncontrollably into the ceiling. The slow $\epsilon$ decay ($0.99995$) paired with the massive `512` node hidden layer helps the agent gradually associate the upper boundary penalty with the action of flapping too near the top.
*   **Sparse Rewards & Stability:** Successfully navigating a pipe early on is extremely rare, making positive reinforcement sparse. 
    *   **Solution**: By implementing a large `100,000` capacity Replay Memory, rare successes are stored and sampled repeatedly during `mini_batch` optimization (batch size of 32). This effectively simulates a prioritized recall, ensuring the agent doesn't "forget" the sequence of actions that led to passing a pipe while it gets flooded with state-data of it crashing.
    *   Additionally, the combination of **Double DQN** (reducing over-optimistic Q-values) and **Dueling architecture** ensures that the agent learns that simply "falling" is functionally okay as long as it isn't near the ground, reserving "flapping" for necessary altitude corrections.
