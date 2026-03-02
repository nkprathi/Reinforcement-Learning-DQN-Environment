import gymnasium as gym
import flappy_bird_gymnasium

env = gym.make("FlappyBird-v0", use_lidar=False)
print("Observation Space:", env.observation_space)
print("Sample Observation:", env.reset()[0])
