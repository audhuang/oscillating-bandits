import numpy as np
import random
import cvxpy as cp
from bandit import Bandit 

class StructuredBandit(): 
    def __init__(self, models, n, a, task=None): 
        self.models = models
        self.n = n
        self.a = a
        self.num_arms = models.shape[1]
        self.num_models = models.shape[0]
        self.counts = np.zeros([self.num_arms])
        self.means = np.zeros([self.num_arms])
        if task is None: 
            self.task = np.random.choice(self.num_models)
        else: 
            self.task = task
        self.bandit = Bandit(self.models[self.task])

    def C_theta(self, theta): 
        i_star = np.argmax(theta)
        i_list = []
        for m in range(len(self.models)): 
            j_star = np.argmax(self.models[m])
            if i_star != j_star:
                i_list.append(np.square(self.models[m] - theta))
        return 0.5 * np.array(i_list)

    def get_alpha(self, theta): 
        d_theta = np.max(theta) - theta
        i_list = self.C_theta(theta)

        alpha = cp.Variable(self.num_arms, integer=True)
        objective = cp.Minimize(d_theta*alpha)
        constraints = [
                    alpha >= 0, 
                    i_list*alpha >= 1,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        #print(prob.value)
        return np.array(alpha.value)


    def get_constraints(self, theta): 
        d_theta = np.max(theta) - theta
        i_list = self.C_theta(theta)

        alpha = cp.Variable(self.num_arms)
        objective = cp.Minimize(0)
        constraints = [
                    alpha >= 0, 
                    i_list*alpha >= 1,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return np.array(alpha.value)


    def run(self, T_list, method=None):
        counts = np.zeros([self.num_arms])
        totals = np.zeros([self.num_arms])
        regrets = []
        # means = np.zeros([num_arms])
        n_e = 0
        temp = [0, 0, 0]
        counts_all = np.zeros([max(T_list), self.num_arms])
        exps = np.zeros([self.num_models])

        for t in range(self.n): 
            if t < self.num_arms:
                reward = self.bandit.pull_arm(t)
                exps += np.square(self.models[:, t] - reward)
                totals[t] += reward
                counts[t] += 1
            else: 
                if method == 'mle': 
                    # average = np.divide(totals, counts)
                    # mle = np.argmin(np.linalg.norm(self.models - average, axis=1))
                    # estimate = self.models[mle]
                    # exps = []
                    # for m in range(num_modes): 
                    #     mode = models[m] 
                    #     means = mode[arms[:t]]
                    #     exp = np.sum(np.square(rewards[:t] - means))
                    #     exps.append(exp)
                    # estimate = models[np.argmin(exps)]
                    estimate = self.models[np.argmin(exps)]
                else: 
                    estimate = np.divide(totals, counts)
                
                # line 6
                i_list = self.C_theta(estimate)
                test = counts/(self.a*np.log(t))
                if np.all(np.dot(i_list, test) >= 1):
                    arm = np.argmax(estimate) 
                    temp[0] += 1
                # line 9
                else: 
                    # line 10
                    if np.min(counts) < (n_e / (2 * self.num_arms)): # (np.sqrt(n_e)/ (2*self.num_arms)):
                        arm = np.argmin(counts)
                        temp[1] += 1
                    # line 12
                    else: 
                        alpha = self.get_alpha(estimate)
                        if alpha.size == self.num_arms: 
                            indices = np.where(test < alpha)
                            arm = random.choice(indices)[0]
                        else: 
                            # constraints = self.get_constraints(estimate)
                            # if constraints.size == self.num_arms: 
                            #     constraints = np.round(constraints)
                            #     indices = np.where(test < constraints)
                            #     arm = random.choice(indices)[0]
                            # else: 
                            #     arm = random.choice(range(self.num_arms))
                            arm = random.choice(range(self.num_arms))
                        temp[2] += 1
                    n_e += 1
                
                reward = self.bandit.pull_arm(arm)
                exps += np.square(self.models[:, arm] - reward)
                totals[arm] += reward
                counts[arm] += 1
            if (t+1) in T_list: 
                regret = self.bandit.calculate_regret(counts) # keep track of regrets
                regrets.append(regret)
            counts_all[t, :] = counts
        # print('estimate: ', estimate)
        # print('counts: ', counts)
        # print('algo: ', temp)
        # regret = self.bandit.calculate_regret(counts)
        # return regret, counts
        return regrets, np.asarray(counts_all)