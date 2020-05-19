import numpy as np
from bandit import Bandit 

class mUCB(): 
    def __init__(self, models, n, alpha_eps, task=None): 
        self.models = models
        self.n = n
        self.delta = 1. / self.n
        self.alpha_eps = alpha_eps
        self.num_arms = models.shape[1]
        self.num_models = models.shape[0]
        self.counts = np.zeros([self.num_arms])
        self.means = np.zeros([self.num_arms])
        if task is None: 
            self.task = np.random.choice(self.num_models)
        else: 
            self.task = task
        self.bandit = Bandit(self.models[self.task])
    
    # calculate confidence epsilon_i,t
    def calculate_e(self): 
        eps = np.sqrt(np.log(self.num_models * self.n**self.alpha_eps / self.delta) / (2*self.counts))
        return eps
    
    # calculate set of compatible models Theta_t
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
    
    # get arm with highest reward
    def get_arm_t(self, Theta_t): 
        index = np.unravel_index(Theta_t.argmax(), Theta_t.shape)
        return index[1]
    

    # update empirical estimates
    def update(self, arm_t, r): 
        self.counts[arm_t] += 1
        n = self.counts[arm_t]
        value = self.means[arm_t]
        self.means[arm_t] = ((n-1)/n) * value + (1./n) * r

    def run(self, T_list): 
        regrets = []
        counts = np.zeros([max(T_list), self.num_arms])
        
        # pull each arm once 
        for t in range(self.num_arms): 
            r = self.bandit.pull_arm(t) 
            self.update(t, r)
            counts[t] = self.counts

        # loop through mUCB algorithm 
        for t in range(self.num_arms, self.n):
            eps = self.calculate_e() # calculate confidence
            Theta_t = self.get_Theta_t(eps) # get compaitble models 
            if len(Theta_t) == 0: 
                return False, np.asarray(counts)
            else: 
                arm_t = self.get_arm_t(Theta_t) # get arm with maximum reward 
            r = self.bandit.pull_arm(arm_t) # pull the arm 
            self.update(arm_t, r)

            if (t+1) in T_list: 
                regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
                regrets.append(regret)
            counts[t, :] = self.counts
        return regrets, np.asarray(counts)

