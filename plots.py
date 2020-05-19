import numpy as np
import matplotlib.pyplot as plt 
import ipdb 

if __name__ == '__main__':

    x = 1000

    

   
    models = [
    [0.5 + 1./np.sqrt(x), 0], 
    [0.5 - 1./np.sqrt(x), 0.5]]
    models = np.array(models)

    num_models, num_arms = models.shape 
    repeats = 20 
    names = ['UCB', 'mUCB', 'structured'] 
    T_list = [500, 1000, 5000, 10000, 20000]

    results = {'structured': np.array([
        [ 248.79945945,  264.21651997,  300.89849156,  300.89849156, 300.89849156],
        [ 148.5321817 ,  290.39195753, 1372.42850451, 2635.63193864, 5162.19692079]]), 
    'UCB': np.array([
        [ 332.26423538,  384.36326748,  513.5476022 ,  560.33040654, 627.31487639],
       [ 140.87946976,  278.56503908, 1009.24091524, 1757.34094131,
        2674.08523499]]), 
    'mUCB': np.array([[  10.63245553,   10.63245553,   10.63245553,   10.63245553,
          10.63245553],
       [ 309.04939573,  609.46577344, 2365.60504924, 2567.04213619,
        2567.04213619]])
    }
    # import pdb; pdb.set_trace()

    plt.figure()
    for name in names: 
        plt.plot(T_list, np.sum(results[name], axis=0) / (num_models*repeats), label=name)
    plt.legend()
    plt.ylabel('regret')
    plt.xlabel('T')
    # plt.show()
    plt.savefig('./plots/tasks_ave')
    plt.close()

    for task in range(num_models): 
        plt.figure()

        for name in names: 
            
            plt.plot(T_list, results[name][task, :] / repeats, label=name)
        plt.legend()
        plt.ylabel('regret')
        plt.xlabel('T')
        plt.savefig('./plots/task_' + str(task))
        plt.close()

    # np.save('./data/pulls', total_pulls)
    # np.save('./data/pulls', total_pulls)


    for task in range(num_models): 
        colors={'UCB' : 'red', 
        'mUCB' : 'green', 
        'structured' : 'blue', 
        }
        
        plt.figure()
        for name in names: 
            plt.plot(T_list, counts[name][task, :, 0] / repeats, color=colors[name], label=name)
            plt.plot(T_list, results[name][task, :, 1] / repeats, '--', color=colors[name])
        plt.legend()
        plt.ylabel('pulls')
        plt.xlabel('T')
        plt.savefig('./plots/pulls_' + str(task))
        plt.close()