import numpy as np
import matplotlib.pyplot as plt 
from mucb import mUCB 
from ucb import UCB
from hypo_test import HypoTest


if __name__ == '__main__':
    n = 500

    # mUCB disadvantageous
    # models = [[2, 0.99], # 1.01
    #           [2 - 1./np.sqrt(n), 1.99]]

   
    # worst case UCB
    # models = [
    # [2., 2 - 1./np.sqrt(n)], 
    # [1 - 1./np.sqrt(n), 1]]

    # only one arm interesting 
    # models = [
    # [0.5, 0.4],
    # [0.5, 0.9]]

    # exists Delta bigger than gap
    models = [
    [5., 4 + 1./np.sqrt(n)], 
    [3.5, 3.5 + 1./np.sqrt(n)]]
    models = np.asarray(models) 
    print(models)

    total = 0
    task = 1
    alpha = 1./n 
    beta = 1./n
    arm = 1
    repeats = 100
    num_models, num_arms = np.shape(models)

    for name in ['UCB', 'mUCB', 'HypoTest']: 
        print('------------%s------------' % name)
        for task in range(num_models): 
            if name == 'UCB': 
                total_regret = 0
                total_pulls = np.zeros([num_arms])
                for _ in range(repeats): 
                    algo = UCB(models, n, task=task) 
                    algo.run()
                    regret = algo.bandit.calculate_regret(algo.counts)
                    total_regret += regret
                    total_pulls += algo.counts
                print('task: %s | regret: %s | pulls: ' % (task, total_regret/repeats), total_pulls/repeats)
            elif name == 'mUCB': 
                total_regret = 0
                total_pulls = np.zeros([num_arms])
                for _ in range(repeats): 
                    algo = mUCB(models, n, task=task) 
                    algo.run()
                    regret = algo.bandit.calculate_regret(algo.counts)
                    total_regret += regret
                    total_pulls += algo.counts
                print('task: %s | regret: %s | pulls: ' % (task, total_regret/repeats), total_pulls/repeats)
            elif name == 'HypoTest': 
                for arm in range(num_arms): 
                    total_regret = 0
                    total_pulls = np.zeros([num_arms])
                    for _ in range(repeats): 
                        algo = HypoTest(models, n, alpha, beta, task=task)
                        algo.run(arm)
                        regret = algo.bandit.calculate_regret(algo.counts)
                        total_regret += regret
                        total_pulls += algo.counts
                    print('task: %s | arm: %s | regret: %s | pulls: ' % (task, arm, total_regret/repeats), total_pulls/repeats)
            
        print('\n')
