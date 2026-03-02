<a name="readme-top"></a>

# Reinforcement Learning: DQN Bird Environment
This repository contains my implementation of a Deep Q-Network (DQN) agent designed to master the Flappy Bird environment.

Project Overview
The core logic of this DQN agent is based on the tutorial series by JohnnyCode (@johnnycode). I have adapted the scratch-built PyTorch implementation to explore how neural networks can approximate the optimal action-value function in a high-speed, discrete action space.

# Algorithm: Deep Q-Learning (DQN)

# Framework: PyTorch / Python

Environment: Flappy Bird (Gym/Gymnasium)

Acknowledgments:
Special thanks to JohnnyCode for the excellent Deep Q-Learning tutorial series. This project was built while following his guide on coding RL algorithms from scratch.

<a href='https://youtu.be/arR7KzlYs4w&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi'><img src='https://img.youtube.com/vi/arR7KzlYs4w/0.jpg' width='400' alt='Install FlappyBird Gymnasium'/></a>

## 1.The Deep Q-Network (DQN) Architecture
The "brain" of the agent is a standard Deep Neural Network consisting of fully connected layers. In this architecture, the input layer accepts the environmental state (the relative positions of the bird and the upcoming pipes), while the output layer predicts the expected Q-values for each possible action. By identifying the action with the highest Q-value for any given state, the agent determines the optimal move to maximize its long-term rewards.

## 2. Training Stability: Experience Replay & YAML Settings
To ensure the model learns generalized patterns rather than just memorizing recent moves, I utilized Experience Replay. This technique involves saving "experiences" (state-action-reward-state sequences) and training the network on random samples from this history. This diversity in training data is crucial for reinforcement learning stability. For better project organization, I decoupled the model logic from the training variables by using a YAML file to load hyperparameters like learning rates and Epsilon values.

## 3. Implement Epsilon-Greedy & Debug the Training Loop
Exploration Strategy and Training OptimizationTo balance the agent's learning, I implemented the $\epsilon$-greedy (Epsilon-Greedy) algorithm. This strategy manages the trade-off between exploration (attempting random maneuvers to discover new strategies) and exploitation (utilizing the model's current knowledge to maximize scores). 

Annealing Process: We initialize $\epsilon$ at 1.0 to ensure the bird starts with purely stochastic behavior. Over time, this value is decayed, gradually shifting the agent’s reliance toward its learned policy. 

Hardware Acceleration: To optimize performance, all environmental states are transformed into PyTorch Tensors. This allows the training loop to leverage CUDA-enabled GPUs, significantly accelerating the backpropagation process compared to CPU-only training.

## 5. Decoupling Learning with a Target Network
A common challenge in Deep Q-Learning is the "moving target" problem, where the values we are trying to predict change every time we update the model. To solve this, I instantiated two identical architectures: the Policy Network and the Target Network.

The Policy Network: This is the primary model that actively interacts with the Flappy Bird environment and undergoes continuous weight updates during the optimization step.

The Target Network: This network is used exclusively to generate the "ground truth" labels (target Q-values). By keeping this network's weights frozen and only synchronizing them with the Policy Network every $N$ steps, we provide a stable objective for the agent. This separation prevents the training from oscillating and ensures smoother mathematical convergence.

## 6. Optimization Process: Understanding the Loss, Backpropagation, and Gradient Descent

To refine the agent's decision-making, the training loop relies on standard deep learning optimization techniques. We quantify the accuracy of our current policy using a Loss Function (typically Mean Squared Error), which calculates the discrepancy between the predicted Q-values and the target values derived from the Bellman equation.

Gradient Descent: This optimization algorithm calculates the gradient (slope) of the loss function relative to the network's parameters. This provides the mathematical "direction" needed to adjust the weights and biases to reduce error.

Backpropagation: Through this iterative process, the gradients are propagated backward through the network layers. By updating the internal parameters in the direction that minimizes the loss, the agent progressively "learns" to associate specific states with higher-reward actions.

## 7. Computational Efficiency through Vectorization
The initial implementation of the DQN update involved iterating through a batch of experiences one by one. While this approach is intuitive for learning the logic, it is computationally expensive as it fails to leverage modern hardware.

To significantly boost performance, I refactored the target Q-value calculations using Vectorized Operations in PyTorch. By treating the entire experience batch as a single multi-dimensional tensor, we utilize PyTorch’s optimized C++/CUDA backends. This parallel processing approach dramatically reduces the time per training step, allowing for faster convergence and more efficient use of GPU resources.

## 8. Validation: Testing the DQN on the CartPole-v1 Environment
Reinforcement Learning models are notoriously sensitive to hyperparameter tuning and small implementation errors. To ensure the core DQN logic was robust and bug-free, I first validated the algorithm on the Gymnasium CartPole-v1 environment.

Because CartPole has a lower-dimensional state space and reaches a solution quickly, it serves as an ideal "unit test" for the agent's learning capabilities. Successfully solving this environment provided the necessary confidence to transition the architecture toward the more complex Flappy Bird task.

## 9. Final Implementation: Training on Flappy Bird
With the algorithm validated, the final phase involved deploying the DQN to master the Flappy Bird environment.

Performance Observations: After an intensive 24-hour training session, the agent successfully learned to navigate through multiple obstacles.

Training Challenges: While the bird achieved high proficiency, reaching a state of "infinite flight" would likely require several days of continuous training. This extended timeline is a characteristic of DQN training, largely due to the high variance in rewards and the complexity of the pixel-based or coordinate-based state transitions in this specific environment.

## 10. Enhancing Stability: Double DQN (DDQN)
Since the original DQN paper, several architectural improvements have been introduced to address its limitations. Double DQN (DDQN) was a primary milestone in this evolution.

The Core Issue: Standard DQN often suffers from overestimation bias, where the agent assigns unrealistically high Q-values to certain actions, leading to inefficient exploration of suboptimal paths.

The Solution: DDQN decouples the action selection from the action evaluation. By using the Policy Network to select the best action and the Target Network to evaluate its value, the agent avoids "wasted" training time on paths that do not yield high rewards. While DDQN provides a more stable learning curve in Flappy Bird, its performance gains can vary depending on the specific environmental stochasticity.

## 11. Structural Optimization: Dueling DQN Architecture
The Dueling Network Architecture is a further refinement designed to accelerate training efficiency without changing the underlying reinforcement learning algorithm.

Decomposition of Q-Values: Instead of estimating a single Q-value for each action, this architecture splits the network's output into two separate streams:
1. State-Value ($V$): The value of being in a specific state.
2. Advantage ($A$): The relative importance of each action compared to others in that state.Benefits: By combining these two streams at the final layer, the model learns which states are inherently valuable, regardless of which action is taken. This is particularly useful in environments like Flappy Bird, where many actions (e.g., not flapping when far from a pipe) have no immediate impact on the outcome. I have integrated this Dueling logic directly into the modular DQN component of this project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
