import numpy as np
from bandit import Bandit 
from scipy.stats import norm


class HypoTest(): 
    def __init__(self, models, n, alpha, beta, task=None): 
        self.models = models
        self.n = n
        self.num_arms = models.shape[1]
        self.num_models = models.shape[0]
        self.counts = np.zeros([self.num_arms])
        self.means = np.zeros([self.num_arms])
        if task is None: 
            self.task = np.random.choice(self.num_models)
        else: 
            self.task = task
        self.bandit = Bandit(self.models[self.task])

        self.alpha = alpha
        self.beta = beta 

    def get_num_pulls(self, arm): 
        num = norm.ppf(self.beta) - norm.ppf(1.- self.alpha)
        den = np.abs(self.models[0, arm] - self.models[1, arm])
        return (num / den)**2

    def calculate_c(self, arm, num_pulls): 
        c = np.min(self.models[:, arm]) + 1. / np.sqrt(num_pulls)*norm.ppf(1.-self.alpha)
        return c

    def calculate_power(self, arm, num_pulls, c): 
        beta = norm.cdf(np.sqrt(num_pulls)*(c - self.models[1, arm]))
        return beta

    def update(self, arm_t, r): 
        self.counts[arm_t] += 1
        n = self.counts[arm_t]
        value = self.means[arm_t]
        self.means[arm_t] = ((n-1)/n) * value + (1./n) * r


    def run(self, arm): 
        num_pulls = np.ceil(self.get_num_pulls(arm))
        c = self.calculate_c(arm, num_pulls) 
        action = arm 
        total = 0
        for t in range(self.n): 
            if t < num_pulls: 
                reward = self.bandit.pull_arm(action)
                total += reward
            elif t == num_pulls: 
                average = total / num_pulls
                # print(average)
                # import ipdb; ipdb.set_trace()
                if average > c: 
                    action = np.argmax(self.models[:, arm])
                else: 
                    action = np.argmin(self.models[:, arm])
                reward = self.bandit.pull_arm(action)
            else: 
                reward = self.bandit.pull_arm(action)
            self.update(action, reward)


