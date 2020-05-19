

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import random
import os 
import pickle

from mucb import mUCB 
from ucb import UCB
from structured_bandit import StructuredBandit
from optim_structured_bandit import OptimStructuredBandit
from dist_structured_bandit import DistStructuredBandit

# class Bandit(): 
#     def __init__(self, mean, dist='normal', var=1): 
#         self.mean = mean
#         self.n_arms = len(mean) 
#         self.dist = dist
#         self.var = var 
#         self.opt_arm = np.argmax(self.mean)

#     def pull_arm(self, arm): 
#         if self.dist == 'normal': 
#             return np.random.normal(self.mean[arm], self.var)
#         else: 
#             print('must specify distribution')

#     def calculate_regret(self, counts): 
#         regret = 0
#         for i in range(self.n_arms): 
#             regret += counts[i] * (self.mean[self.opt_arm] - self.mean[i])
#         return regret

# class UCB(): 
#     def __init__(self, models, n, alpha_ucb, task=None): 
#         self.models = models
#         self.n = n
#         self.alpha = alpha_ucb
#         self.num_arms = models.shape[1]
#         self.num_models = models.shape[0]
#         self.counts = np.zeros([self.num_arms])
#         self.means = np.zeros([self.num_arms])
#         if task is None: 
#             self.task = np.random.choice(self.num_models)
#         else: 
#             self.task = task
#         self.bandit = Bandit(self.models[self.task])

#     def select_arm_ucb(self, t): 
#         for a in range(self.num_arms): 
#             if self.counts[a] == 0: 
#                 return a
#         ucb_values = self.means + np.sqrt((self.alpha*np.log(t)) / self.counts)
#         return np.argmax(ucb_values)

#     def update_arm(self, action, reward): 
#         self.counts[action] += 1

#         n = self.counts[action]
#         value = self.means[action]
#         self.means[action] = ((n-1)/n) * value + (1./n) * reward

#     def run(self, T_list): 
#         regrets = []
#         counts = np.zeros([max(T_list), self.num_arms])
#         for t in range(self.n): 
#             action = self.select_arm_ucb(t)
#             reward = self.bandit.pull_arm(action)
#             self.update_arm(action, reward)
#             if (t+1) in T_list: 
#                 regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
#                 regrets.append(regret)
#             counts[t, :] = self.counts

#         return regrets, np.asarray(counts)

# class mUCB(): 
#     def __init__(self, models, n, alpha_eps, task=None): 
#         self.models = models
#         self.n = n
#         self.delta = 1. / self.n
#         self.alpha_eps = alpha_eps
#         self.num_arms = models.shape[1]
#         self.num_models = models.shape[0]
#         self.counts = np.zeros([self.num_arms])
#         self.means = np.zeros([self.num_arms])
#         if task is None: 
#             self.task = np.random.choice(self.num_models)
#         else: 
#             self.task = task
#         self.bandit = Bandit(self.models[self.task])
    
#     # calculate confidence epsilon_i,t
#     def calculate_e(self): 
#         eps = np.sqrt(np.log(self.num_models * self.n**self.alpha_eps / self.delta) / (2*self.counts))
#         return eps
    
#     # calculate set of compatible models Theta_t
#     def get_Theta_t(self, eps): 
#         indices = []
#         for i in range(self.num_models): 
#             model = self.models[i]
#             check = True
#             for j in range(self.num_arms): 
#                 if np.abs(model[j] - self.means[j]) > eps[j]: 
#                     check = False 
#             if check == True: 
#                 indices.append(i)
#         return self.models[indices, :]
    
#     # get arm with highest reward
#     def get_arm_t(self, Theta_t): 
#         index = np.unravel_index(Theta_t.argmax(), Theta_t.shape)
#         return index[1]
    

#     # update empirical estimates
#     def update(self, arm_t, r): 
#         self.counts[arm_t] += 1
#         n = self.counts[arm_t]
#         value = self.means[arm_t]
#         self.means[arm_t] = ((n-1)/n) * value + (1./n) * r

#     def run(self, T_list): 
#         regrets = []
#         counts = np.zeros([max(T_list), self.num_arms])
        
#         # pull each arm once 
#         for t in range(self.num_arms): 
#             r = self.bandit.pull_arm(t) 
#             self.update(t, r)
#             counts[t] = self.counts

#         # loop through mUCB algorithm 
#         for t in range(self.num_arms, self.n):
#             eps = self.calculate_e() # calculate confidence
#             Theta_t = self.get_Theta_t(eps) # get compaitble models 
#             if len(Theta_t) == 0: 
#             	return False, np.asarray(counts)
#             else: 
#             	arm_t = self.get_arm_t(Theta_t) # get arm with maximum reward 
#             r = self.bandit.pull_arm(arm_t) # pull the arm 
#             self.update(arm_t, r)

#             if (t+1) in T_list: 
#                 regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
#                 regrets.append(regret)
#             counts[t, :] = self.counts
#         return regrets, np.asarray(counts)


# class OptimStructuredBandit(): 
#     def __init__(self, models, n, a, alpha_eps, task=None): 
#         self.models = models
#         self.n = n
#         self.delta = 1./n # mUCB parameter
#         self.a = a # structured bandits parameter
#         self.alpha_eps = alpha_eps # mUCB parameter 
#         self.num_arms = models.shape[1]
#         self.num_models = models.shape[0]
#         self.counts = np.zeros([self.num_arms])
#         self.means = np.zeros([self.num_arms])
#         self.totals = np.zeros([self.num_arms])
#         if task is None: 
#             self.task = np.random.choice(self.num_models)
#         else: 
#             self.task = task
#         self.bandit = Bandit(self.models[self.task])

        

#         # pre-calculate to reduce time 
#         self.C_theta_all = np.array([self.C_theta(theta) for theta in self.models])
#         self.alpha_all = np.array([self.get_alpha(theta, [theta], self.n) for theta in self.models])
            

#     # returns KL divergence of all models with different i* than theta
#     def C_theta(self, theta): 
#         i_star = np.argmax(theta)
#         i_list = []
#         for m in range(len(self.models)): 
#             j_star = np.argmax(self.models[m])
#             if i_star != j_star:
#                 i_list.append(np.square(self.models[m] - theta))
#         return 0.5 * np.array(i_list)

#     # solves optimization problem to get number of pulls
#     def get_alpha(self, theta, Theta_t, T): 
#         d_theta = np.max(Theta_t, axis=1).reshape([len(Theta_t), 1]) - Theta_t
#         i_list = self.C_theta(theta)

#         alpha = cp.Variable(self.num_arms, integer=True)
#         objective = cp.Minimize(cp.sum(d_theta@alpha))
#         constraints = [
#                     alpha >= 0, 
#                     i_list*alpha >= 1,
#                     # cp.sum(alpha) == T,
#         ]
#         prob = cp.Problem(objective, constraints)
#         prob.solve()
#         return np.array(alpha.value)


#     # get list of compatible models (from mUCB algorithm)
#     def get_Theta_t(self, eps): 
#         indices = []
#         self.means = np.divide(self.totals, self.counts)
#         for i in range(self.num_models): 
#             model = self.models[i]
#             if np.all(np.abs(model - self.means) <= eps): 
#                 indices.append(i)
#         return self.models[indices, :], indices
    
#     # calculate epsilon (from mUCB algorithm), alpha_eps is hyperparam > 0
#     def calculate_e(self): 
#         eps = np.sqrt(np.log(self.num_models * self.n**self.alpha_eps / self.delta) / (2*self.counts))
#         return eps


#     def run(self, T_list, print_stuff=False):
#         regrets = []
#         counts_all = np.zeros([max(T_list), self.num_arms])

#         for t in range(self.n): 
#             # pull each arm once 
#             if t < self.num_arms:
#                 reward = self.bandit.pull_arm(t)
#                 self.totals[t] += reward
#                 self.counts[t] += 1
#             else: 
#                 confident = False
#                 eps = self.calculate_e() # calculate confidence
#                 Theta_t, indices = self.get_Theta_t(eps) # get compatible models 
#                 if Theta_t.size == 0: # quit if Theta_t is empty
#                     return False, self.counts

#                 # get index of optimistic model and optimistic model 
#                 optimistic_idx = np.unravel_index(Theta_t.argmax(), Theta_t.shape)[0]
#                 optimistic_theta = Theta_t[optimistic_idx]
                
#                 # get KL divergences
#                 i_list = self.C_theta(optimistic_theta)  
#                 test = self.counts/(self.a*np.log(t))
                
#                 # check if we are confident enough; then pull best arm
#                 # I just added this Theta_t condition... 
#                 if np.all(np.dot(i_list, test) >= 1) or len(Theta_t) == 1:
#                     arm = np.argmax(optimistic_theta) 
#                     confident = True
                
#                 # otherwise, compute alpha and pull arm which is < alpha
#                 else: 
#                     # this is to save computation time if there is only one 
#                     # model in Theta_t by utilizing some pre-computed alphas 
#                     if len(Theta_t) == 1: 
#                         alpha = self.alpha_all[indices]
#                         idxs = np.where(test < np.squeeze(alpha))
#                         arm = random.choice(idxs[0])
#                     else: 
#                         # compute alpha and pull arm
#                         alpha = self.get_alpha(optimistic_theta, Theta_t, self.n)
#                         if alpha.size == self.num_arms: 
#                             idxs = np.where(test < alpha)
#                             # if len(idxs[0]) > 1: 
#                             #     import ipdb; ipdb.set_trace()
#                             arm = random.choice(idxs[0])                                
#                         else: 
#                             arm = random.choice(range(self.num_arms))
                
#                 reward = self.bandit.pull_arm(arm)
#                 self.totals[arm] += reward
#                 self.counts[arm] += 1

#                 if print_stuff: 
#                     # print('means: ', self.means, 'diff: ', self.models - self.means)
#                     # print('eps: ', eps)
#                     print('Theta_t: ', indices)
#                     if confident == False: 
#                         print('test: ', test)
#                         print('alpha: ', alpha)
#                     print('indices: ', idxs, 'arm: ', arm, ' counts: ', self.counts)
#                     print('\n')
            
#             if (t+1) in T_list: 
#                 regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
#                 regrets.append(regret)
#             counts_all[t, :] = self.counts
#         return regrets, np.asarray(counts_all)


# class DistStructuredBandit(): 
#     def __init__(self, models, n, alpha_eps, task=None): 
#         self.models = models
#         self.n = n
#         self.delta = 1./n # mUCB parameter
#         self.alpha_eps = alpha_eps # mUCB parameter 
#         self.num_arms = models.shape[1]
#         self.num_models = models.shape[0]
#         self.counts = np.zeros([self.num_arms])
#         self.means = np.zeros([self.num_arms])
#         self.totals = np.zeros([self.num_arms])
#         if task is None: 
#             self.task = np.random.choice(self.num_models)
#         else: 
#             self.task = task
#         self.bandit = Bandit(self.models[self.task])

        

#         # pre-calculate to reduce time 
#         self.C_theta_all = np.array([self.C_theta(theta) for theta in self.models])
#         self.alpha_all = np.array([self.get_alpha(theta, [theta], self.n) for theta in self.models])
            

#     # returns KL divergence of all models with different i* than theta
#     def C_theta(self, theta): 
#         i_star = np.argmax(theta)
#         i_list = []
#         for m in range(len(self.models)): 
#             j_star = np.argmax(self.models[m])
#             if i_star != j_star:
#                 i_list.append(np.square(self.models[m] - theta))
#         return 0.5 * np.array(i_list)

#     # solves optimization problem to get number of pulls
#     def get_alpha(self, theta, Theta_t, T): 
#         d_theta = np.max(Theta_t, axis=1).reshape([len(Theta_t), 1]) - Theta_t
#         i_list = self.C_theta(theta)

#         alpha = cp.Variable(self.num_arms, integer=True)
#         objective = cp.Minimize(cp.sum(d_theta@alpha))
#         constraints = [
#                     alpha >= 0, 
#                     i_list*alpha >= 1,
#                     # cp.sum(alpha) == T,
#         ]
#         prob = cp.Problem(objective, constraints)
#         prob.solve()
#         return np.array(alpha.value)


#     # get list of compatible models (from mUCB algorithm)
#     def get_Theta_t(self, eps): 
#         indices = []
#         self.means = np.divide(self.totals, self.counts)
#         for i in range(self.num_models): 
#             model = self.models[i]
#             if np.all(np.abs(model - self.means) <= eps): 
#                 indices.append(i)
#         return self.models[indices, :], indices
    
#     # calculate epsilon (from mUCB algorithm), alpha_eps is hyperparam > 0
#     def calculate_e(self): 
#         eps = np.sqrt(np.log(self.num_models * self.n**self.alpha_eps / self.delta) / (2*self.counts))
#         return eps


#     def run(self, T_list, print_stuff=False):
#         regrets = []
#         counts_all = np.zeros([max(T_list), self.num_arms])

#         for t in range(self.n): 
#             # pull each arm once 
#             if t < self.num_arms:
#                 reward = self.bandit.pull_arm(t)
#                 self.totals[t] += reward
#                 self.counts[t] += 1
#             else: 
#                 eps = self.calculate_e() # calculate confidence
#                 Theta_t, indices = self.get_Theta_t(eps) # get compatible models 
#                 if Theta_t.size == 0: # quit if Theta_t is empty
#                     return False, self.counts
                
#                 # if only one mode in Theta_t, play optimal arm 
#                 if len(Theta_t) == 1: 
#                     arm = np.argmax(np.squeeze(Theta_t))
#                 # otherwise, use result of optimization problem as guiding distribution
#                 else: 
#                     # get index of optimistic model and optimistic model 
#                     optimistic_idx = np.unravel_index(Theta_t.argmax(), Theta_t.shape)[0]
#                     optimistic_theta = Theta_t[optimistic_idx]

#                     # use alpha as distribution over arms 
#                     alpha = self.get_alpha(optimistic_theta, Theta_t, self.n)
#                     masses = alpha / np.sum(alpha) * self.n
#                     idxs = np.where(self.counts < masses)
#                     arm = random.choice(idxs[0])   
                
#                 reward = self.bandit.pull_arm(arm)
#                 self.totals[arm] += reward
#                 self.counts[arm] += 1

#                 if print_stuff: 
#                     # print('means: ', self.means, 'diff: ', self.models - self.means)
#                     # print('eps: ', eps)
#                     print('Theta_t: ', indices)
#                     print('alpha:\t', alpha)
#                     print('masses:\t', masses)
#                     print('indices: ', idxs, 'arm: ', arm)
#                     print('counts: ', self.counts)
#                     print('\n')
            
#             if (t+1) in T_list: 
#                 regret = self.bandit.calculate_regret(self.counts) # keep track of regrets
#                 regrets.append(regret)
#             counts_all[t, :] = self.counts
#         return regrets, np.asarray(counts_all)



# models = [
#     [0.9, 0.75, 0.45, 0.55, 0.58, 0.61, 0.65],
#     [0.75, .89, .45, .55, .58, .61, .65], 
#     [.2, .23, .45, .35, .3, .18, .25],
#     [.34, .31, .45, .725, .33, .37, .47], 
#     [.6, .5, .45, .35, .95, .9, .8],
#     ]

# c = 10000
# models = [
#           [0.5 + 1./np.sqrt(c), 0], 
#           [0.5 - 1./np.sqrt(c), 0.5],
# ]

# models = [
# [0.75, 0.75 - 1./np.sqrt(c)], 
# [0.25, 0.25 + 1./np.sqrt(c)]]

# models = np.array(models)
if __name__ == '__main__':
    runs = 10
    num_models = 3
    num_arms = 5
    repeats = 20

    names = ['UCB', 'mUCB', 'OptimStructured', 'DistStructured'] 
    T_list = [500, 1000, 5000, 10000] # list of timesteps to record 
    n = max(T_list) # total run time 



    for run in range(runs): 
        folder_name = 'size_' + str(num_models) + '_' + str(num_arms) + '/'
        subfolder_name = 'size_' + str(num_models) + '_' + str(num_arms) + '_run_' + str(run) + '/'
        dirname = './plots/' + folder_name + subfolder_name
        if not os.path.exists('./plots/' + folder_name): 
            os.mkdir('./plots/' + folder_name)
        if not os.path.exists(dirname): 
            os.mkdir(dirname)
    

        while True:     
            models = np.random.uniform(size=[num_models, num_arms])
            if len(np.unique(np.argmax(models, axis=1))) == min(num_models, num_arms): 
                print(models)
                print(np.argmax(models, axis=1))
                break
        np.savetxt(dirname + "model.csv", models)

        num_models, num_arms = np.shape(models)
        
        results = {}
        total_pulls = {}
        record_til = 500
        for name in names: 
            results[name] = np.zeros([num_models, repeats, len(T_list)])
            total_pulls[name] = np.zeros([num_models, repeats, record_til, num_arms])


        '''hyperparameters'''
        a = 2 # structured bandits hyperparameter for confidence
        alpha_eps = 2 # mUCB hyperparameter for calculating epsilon
        alpha_ucb = 2


        for task in range(num_models)[:]: 
            print('task: ', task)
            for name in names: 
                if name == 'UCB': 
                    for r in range(repeats): 
                        algo = UCB(models, n, alpha_ucb, task=task) 
                        regrets, counts = algo.run(T_list)
                        results[name][task, r] = regrets
                        total_pulls[name][task, r] = counts[:record_til]
                elif name == 'mUCB': 
                    for r in range(repeats): 
                        regrets = False 
                        while regrets == False: 
                            algo = mUCB(models, n, alpha_eps, task=task) 
                            regrets, counts = algo.run(T_list)
                        results[name][task, r] = regrets
                        total_pulls[name][task, r] = counts[:record_til]
                
                elif name == 'OptimStructured': 
                    for r in range(repeats): 
                        regrets = False 
                        while regrets == False: 
                            algo = OptimStructuredBandit(models, n, a, alpha_eps, task=task) 
                            regrets, counts = algo.run(T_list, print_stuff=False)
                        results[name][task, r] = regrets
                        total_pulls[name][task, r] = counts[:record_til]

                elif name == 'DistStructured': 
                    for r in range(repeats): 
                        regrets = False 
                        while regrets == False: 
                            algo = DistStructuredBandit(models, n, alpha_eps, task=task) 
                            regrets, counts = algo.run(T_list, print_stuff=False)
                        results[name][task, r] = regrets
                        total_pulls[name][task, r] = counts[:record_til]
                            
                            
                            
                            
                    
        # print(results)
        with open(dirname + 'results_run_' + str(run) + '.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        # plotting 
        plt.figure()
        for name in names: 
            std = np.std(results[name], axis=(0,1))
            # plt.plot(T_list, np.sum(np.sum(results[name], axis=0), axis=0) / (num_models*repeats), label=name)
            plt.errorbar(T_list, np.sum(np.sum(results[name], axis=0), axis=0) / (num_models*repeats), yerr=std, label=name)
        plt.legend()
        plt.ylabel('regret')
        plt.xlabel('T')
        plt.title('Average')
        plt.savefig(dirname + 'tasks_ave_run_' + str(run))
        # plt.show()
        plt.close()

        for task in range(num_models): 
            plt.figure()
            for name in names: 
                std = np.std(results[name][task], axis=(0))
                # plt.plot(T_list, np.sum(results[name][task], axis=0) / repeats, label=name)
                plt.errorbar(T_list, np.sum(results[name][task], axis=0) / repeats, yerr=std, label=name)
            plt.legend()
            plt.title('Task ' + str(task))
            plt.ylabel('regret')
            plt.xlabel('T')
            plt.savefig(dirname + 'task_' + str(task) + '_run_' + str(run))
            # plt.show()
            plt.close()

        # np.save('./data/pulls', total_pulls)
        # np.save('./data/pulls', total_pulls)


        for task in range(num_models): 
            colors={'UCB' : 'blue', 
            'mUCB' : 'orange', 
            'OptimStructured' : 'green', 
            'DistStructured' : 'red',
            }
            
            
            for arm in range(num_arms): 
                plt.figure()
                for name in names: 
                    std = np.std(total_pulls[name][task, :, :, arm], axis=(0))
                    line = np.sum(total_pulls[name][task, :, :, arm], axis=0) / repeats
                    plt.plot(range(record_til), line, color=colors[name], label=name)
                    plt.fill_between(range(record_til), line - std, line+std, color=colors[name], alpha=0.2)

                    # plt.errorbar(range(record_til), np.sum(total_pulls[name][task, :, :, arm], axis=0) / repeats, yerr=std, color=colors[name], label=name)
                plt.legend()
                plt.ylabel('pulls')
                plt.xlabel('T')
                plt.savefig(dirname + 'pulls_task_' + str(task) + '_arm_' + str(arm) + '_run_' + str(run))
                plt.close()


