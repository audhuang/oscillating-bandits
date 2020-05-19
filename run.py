import numpy as np
import matplotlib.pyplot as plt 
from mucb import mUCB 
from ucb import UCB
from structured_bandit import StructuredBandit
import os
# from hypo_test import HypoTest


if __name__ == '__main__':
    x = 10000

    name = 'mucb_hard'
    dirname = './plots/' + name + '/'
    os.mkdir(dirname)
    prename = dirname + name
    # models = [
    # [0.5 + 1./np.sqrt(x), 0], 
    # [0.5 - 1./np.sqrt(x), 0.5]]

    # models = [
    # [0.75, 0.75 - 1./np.sqrt(x)], 
    # [0.25, 0.25 + 1./np.sqrt(x)]]

    # models = [
    # [0.5 + 1./np.sqrt(n), 0, 0.5 - 1./np.sqrt(n) - np.sqrt(np.log(n))], 
    # [0.5 - 1./np.sqrt(n), 0.5, 0.5 - 1./np.sqrt(n)]]

    # models = [
    # [0.5 + 1./np.sqrt(n), 0], 
    # [0.5 - 1./np.sqrt(n), 0.5], 
    # [-1, 0.]]

    # models from mUCB paper
    # models = [
    # [0.9, 0.75, 0.45, 0.55, 0.58, 0.61, 0.65],
    # [0.75, .89, .45, .55, .58, .61, .65], 
    # [.2, .23, .45, .35, .3, .18, .25],
    # [.34, .31, .45, .725, .33, .37, .47], 
    # [.6, .5, .45, .35, .95, .9, .8],
    # ]

    models = np.asarray(models) 

    a = 1.
    repeats = 10 # repetitions of each experiment
    num_models, num_arms = np.shape(models)

    names = ['UCB', 'mUCB', 'structured'] 
    T_list = [500, 1000, 5000, 10000] # list of timesteps to try
    results = {}
    total_pulls = {}
    for name in names: 
        results[name] = np.zeros([num_models, len(T_list)])
        total_pulls[name] = np.zeros([num_models, max(T_list), num_arms])

    # for i in range(len(T_list)): 
    # n = T_list[i] # time steps 
    n = max(T_list)
    alpha = 1./n 
    beta = 1./n # hyperparams


    
    for task in range(num_models): 
        print('task: ', task)
        for name in names: 
            if name == 'UCB': 
                for _ in range(repeats): 
                    algo = UCB(models, n, task=task) 
                    regrets, counts = algo.run(T_list)
                    results[name][task] += regrets
                    total_pulls[name][task] += counts
            elif name == 'mUCB': 
                for _ in range(repeats): 
                    algo = mUCB(models, n, task=task) 
                    regrets, counts = algo.run(T_list)
                    results[name][task] += regrets
                    total_pulls[name][task] += counts
            elif name == 'structured': 
                for _ in range(repeats): 
                    algo = StructuredBandit(models, n, a, task=task) 
                    regrets, counts = algo.run(T_list, method='mle')
                    results[name][task] += regrets
                    total_pulls[name][task] += counts
                    # print(regrets, counts)
                    # import ipdb; ipdb.set_trace()
            # print(name, total_pulls / repeats)
#             elif name == 'HypoTest': 
#                 # for arm in range(num_arms): 
#                     total_regret = 0
#                     total_pulls = np.zeros([num_arms])
#                     for _ in range(repeats): 
#                         algo = HypoTest(models, n, alpha, beta, task=task)
#                         regrets = algo.run(arm, num_pulls = num_pulls)
#                         results[name + '-'+str(arm+1)][i] = results[name + '-'+str(arm+1)][i] + regrets
                        
                
    print(results)
    # import ipdb; ipdb.set_trace()

    # plotting 
    plt.figure()
    for name in names: 
        plt.plot(T_list, np.sum(results[name], axis=0) / (num_models*repeats), label=name)
    plt.legend()
    plt.ylabel('regret')
    plt.xlabel('T')
    # plt.show()
    plt.savefig(prename + '_tasks_ave')
    plt.close()

    for task in range(num_models): 
        plt.figure()
        for name in names: 
            plt.plot(T_list, results[name][task, :] / repeats, label=name)
        plt.legend()
        plt.ylabel('regret')
        plt.xlabel('T')
        plt.savefig(prename + '_task_' + str(task))
        plt.close()

    # np.save('./data/pulls', total_pulls)
    # np.save('./data/pulls', total_pulls)


    for task in range(num_models): 
        colors={'UCB' : 'red', 
        'mUCB' : 'green', 
        'structured' : 'blue', 
        }
        
        
        for arm in range(num_arms): 
            plt.figure()
            for name in names: 
                plt.plot(range(np.max(T_list)), total_pulls[name][task, :, arm] / repeats, color=colors[name], label=name)
                # plt.plot(range(np.max(T_list)), total_pulls[name][task, :, arm] / repeats, '--', color=colors[name])
            plt.legend()
            plt.ylabel('pulls')
            plt.xlabel('T')
            plt.savefig(prename + '_pulls_task_' + str(task) + '_arm_' + str(arm))
            plt.close()


    import ipdb; ipdb.set_trace()


    # for name in ['UCB', 'mUCB', 'HypoTest']: 
    #     print('------------%s------------' % name)
    #     for task in range(num_models): 
    #         if name == 'UCB': 
    #             total_regret = 0
    #             total_pulls = np.zeros([num_arms])
    #             for _ in range(repeats): 
    #                 algo = UCB(models, n, task=task) 
    #                 algo.run()
    #                 regret = algo.bandit.calculate_regret(algo.counts)
    #                 total_regret += regret
    #                 total_pulls += algo.counts
    #             print('task: %s | regret: %s | pulls: ' % (task, total_regret/repeats), total_pulls/repeats)
    #         elif name == 'mUCB': 
    #             total_regret = 0
    #             total_pulls = np.zeros([num_arms])
    #             for _ in range(repeats): 
    #                 algo = mUCB(models, n, task=task) 
    #                 algo.run()
    #                 regret = algo.bandit.calculate_regret(algo.counts)
    #                 total_regret += regret
    #                 total_pulls += algo.counts
    #             print('task: %s | regret: %s | pulls: ' % (task, total_regret/repeats), total_pulls/repeats)
    #         elif name == 'HypoTest': 
    #             for arm in range(num_arms): 
    #                 total_regret = 0
    #                 total_pulls = np.zeros([num_arms])
    #                 for _ in range(repeats): 
    #                     algo = HypoTest(models, n, alpha, beta, task=task)
    #                     algo.run(arm)
    #                     regret = algo.bandit.calculate_regret(algo.counts)
    #                     total_regret += regret
    #                     total_pulls += algo.counts
    #                 print('task: %s | arm: %s | regret: %s | pulls: ' % (task, arm, total_regret/repeats), total_pulls/repeats)
            
    #     print('\n')
