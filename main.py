import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
 
EPISODES      = 500_000
ALPHA         = 0.01
GAMMA         = 1.0
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = EPISODES * 0.8
 
Q = defaultdict(lambda: np.zeros(2))
 
def get_epsilon(episode):
    return max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY)
 
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return int(np.argmax(Q[state]))
 
def train():
    env = gym.make("Blackjack-v1", sab=True)
    rewards_per_episode = []
 
    for ep in range(EPISODES):
        state, _ = env.reset()
        epsilon = get_epsilon(ep)
        done = False
 
        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
            best_next = np.max(Q[next_state]) if not done else 0.0
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            state = next_state
 
        rewards_per_episode.append(reward)
 
    env.close()
    return rewards_per_episode
 
def evaluate(n_eval=100_000):
    env = gym.make("Blackjack-v1", sab=True)
    wins, losses, pushes = 0, 0, 0
 
    for _ in range(n_eval):
        state, _ = env.reset()
        done = False
 
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
 
        if reward > 0:   wins   += 1
        elif reward < 0: losses += 1
        else:            pushes += 1
 
    env.close()
