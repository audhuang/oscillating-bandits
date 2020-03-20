import numpy as np
from bandit import Bandit 

class mUCB(): 
    def __init__(self, models, n, task=None): 
        self.models = models
        self.n = n
        self.delta = 1. / self.n
        self.num_arms = models.shape[1]
        self.num_models = models.shape[0]
        self.counts = np.zeros([self.num_arms])
        self.means = np.zeros([self.num_arms])
        if task is None: 
            self.task = np.random.choice(self.num_models)
        else: 
            self.task = task
        self.bandit = Bandit(self.models[self.task])
    
    def calculate_e(self): 
        eps = np.sqrt(np.log(self.num_models * self.n**2 / self.delta) / (2*self.counts))
        return eps
    
    def get_Theta_t(self, eps): 
        indices = []
        for i in range(self.num_models): 
            model = self.models[i]
            check = True
            for j in range(self.num_arms): 
                if np.abs(model[j] - self.means[j]) > eps[j]: 
                    check = False 
            if check == True: 
                indices.append(i)
        return self.models[indices, :]
    
    def get_arm_t(self, Theta_t): 
        index = np.unravel_index(Theta_t.argmax(), Theta_t.shape)
        return index[1]
    

    def update(self, arm_t, r): 
        self.counts[arm_t] += 1
        n = self.counts[arm_t]
        value = self.means[arm_t]
        self.means[arm_t] = ((n-1)/n) * value + (1./n) * r

    def run(self): 
        for t in range(self.num_arms): 
            r = self.bandit.pull_arm(t) 
            self.update(t, r)
        # print(self.means)
        for t in range(self.num_arms, self.n):
            eps = self.calculate_e()
            Theta_t = self.get_Theta_t(eps) 
            if len(Theta_t) == 0: 
            	arm_t = np.random.choice(self.num_arms)
            else: 
            	arm_t = self.get_arm_t(Theta_t)
            # print(t, eps, len(Theta_t), arm_t) 
            r = self.bandit.pull_arm(arm_t) 
            self.update(arm_t, r)

