import numpy as np


class SocialMetrics:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.observations = []
        self.rewards = []

        self.utilitarian_eff = None
        self.equality = None
        self.sustainability = None
        self.peace = None
    def add_step(self, obs, rws):
        self.observations.append(obs)
        self.rewards.append(rws)

    def compute_metrics(self):
        self.compute_utilitarian_eff()
        self.compute_equality()
        self.compute_sustainability()
        self.compute_peace()

        self.observations = []
        self.rewards = []
    def compute_utilitarian_eff(self):
        total_rewards = sum(sum(agent_rewards) for agent_rewards in self.rewards)
        self.utilitarian_eff = total_rewards / self.num_agents

    def compute_equality(self):
        rewards = np.array(self.rewards)
        agent_rewards = np.sum(rewards, axis=0)
        gini = gini_coefficient(agent_rewards)
        self.equality = 1 - gini


    def compute_sustainability(self):
        times = 0
        for i in range(len(self.rewards)):
            ts = np.array(self.rewards[i]) * i
            sum_t = np.sum(ts)
            times += sum_t

        self.sustainability = times / len(self.rewards)

    def compute_peace(self):
        total_steps = len(self.observations)
        tagged_steps = 0
        for obs_step in self.observations:
            tagged_steps += sum([1 for o in obs_step if type(o) == type(None)])

        self.peace = (total_steps * self.num_agents - tagged_steps) / self.num_agents

def gini_coefficient(x):
    """
    Calculate the Gini coefficient of a numpy array.
    :param x: numpy array of rewards.
    :return: Gini coefficient.
    """
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))