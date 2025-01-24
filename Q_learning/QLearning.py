from re import S
from matplotlib import pyplot as plt
import numpy as np
class QLearning:
    def __init__(self, env, alpha, gamma, epsilon, n_episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.Q = np.random.rand(env.map.shape[0], env.map.shape[1], env.actions)

    def epsilon_greedy_policy(self, s, epsilon):
        # Epsilon greedy policy (choose a random action with probability epsilon)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.env.actions)
        else:
            return np.argmax(self.Q[s[0], s[1]])

    def episode(self, alpha, epsilon):
        # Episode execution. Generate an action with epsilon_greedy_policy, call step, appy learning function
        # This function should return the total reward obtained in this episode
        s = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            a = self.epsilon_greedy_policy(s, epsilon)
            s_prime, reward, done = self.env.step(a)
            self.Q[s[0], s[1], a] += alpha * (reward + self.gamma * np.max(self.Q[s_prime[0], s_prime[1]]) - self.Q[s[0], s[1], a])
            s = s_prime
            total_reward += reward
        self.env.reset()
        return total_reward

    def train(self, n_episodes, check_every_n=100):
        """Execute n_episodes and every `check_every_n` episodes print the average reward and store it.
           As the initial position is random, this number will not be super stable..."""
        rewards = []
        avg_rewards = []
        for i in range(n_episodes):
            reward = self.episode(self.alpha, self.epsilon)
            rewards.append(reward)
            if (i+1) % check_every_n == 0:
              avg_reward = np.mean(rewards)
              print(f"Average reward at iteration {i+1}: {avg_reward}")
              avg_rewards.append(avg_reward)
    def get_optimal_policy(self):
        """Once training is done, retrieve the optimal policy from Q(s,a)."""
        policy = np.argmax(self.Q, axis = 2)
        policy[self.env.map == 1] = -1
        return policy

    def value_function(self):
        """Once training is done, retrieve the optimal value function from from Q(s,a)"""
        v = np.max(self.Q, axis = 2)
        v[self.env.map == 1] = -np.inf
        return v
