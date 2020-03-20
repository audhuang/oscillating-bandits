import numpy as np 

class Bandit(): 
    def __init__(self, mean, dist='normal', var=1): 
        self.mean = mean
        self.n_arms = len(mean) 
        self.dist = dist
        self.var = var 
        self.opt_arm = np.argmax(self.mean)

    def pull_arm(self, arm): 
        if self.dist == 'normal': 
            return np.random.normal(self.mean[arm], self.var)
        else: 
            print('must specify distribution')

    def calculate_regret(self, counts): 
        regret = 0
        for i in range(self.n_arms): 
            regret += counts[i] * (self.mean[self.opt_arm] - self.mean[i])
        return regret 