import numpy as np 
from bandit import Bandit 


class UCB(): 
    def __init__(self, models, n, alpha_ucb, task=None): 
        self.models = models
        self.n = n
        self.alpha = alpha_ucb
        self.num_arms = models.shape[1]
        self.num_models = models.shape[0]
        self.counts = np.zeros([self.num_arms])
        self.means = np.zeros([self.num_arms])
        if task is None: 
            self.task = np.random.choice(self.num_models)
        else: 
            self.task = task
        self.bandit = Bandit(self.models[self.task])

    def select_arm_ucb(self, t): 
        for a in range(self.num_arms): 
            if self.counts[a] == 0: 
                return a
        ucb_values = self.means + np.sqrt((self.alpha*np.log(t)) / self.counts)
        return np.argmax(ucb_values)

    def update_arm(self, action, reward): 
        self.counts[action] += 1

        n = self.counts[action]
        value = self.means[action]
        self.means[action] = ((n-1)/n) * value + (1./n) * reward

    def run(self, T_list): 
        regrets = []
        counts = np.zeros([max(T_list), self.num_arms])
        for t in range(self.n): 
            action = self.select_arm_ucb(t)
            reward = self.bandit.pull_arm(action)
            self.update_arm(action, reward)
            if (t+1) in T_list: 
                regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
                regrets.append(regret)
            counts[t, :] = self.counts

        return regrets, np.asarray(counts)