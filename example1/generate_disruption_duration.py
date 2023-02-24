import numpy as np
import math


def prob_mass(num_evals, T_scenario):

    if T_scenario == 'uniform':
        # uniform
        ans = 1/num_evals * np.ones(num_evals)
    
    elif T_scenario == 'exponential':
        # decreasing (exponential)
        ans = np.zeros(num_evals)
        tmp_lambda = 1.4
        for i in range(num_evals):
            ans[i] = tmp_lambda ** (num_evals - i - 1)

    elif T_scenario == 'normal':
        miu = num_evals / 2
        std = num_evals / 4
        
        ans = np.zeros(num_evals)
        for i in range(num_evals):
            ans[i] = math.exp((((i - miu)/std)**2) * (-0.5))

    elif T_scenario == 'Dirac_0':
        ans = np.zeros(num_evals)
        ans[0] = 1
    
    elif T_scenario == 'Dirac_Tub':
        ans = np.zeros(num_evals)
        ans[num_evals - 1] = 1

    elif T_scenario == 'bi_Dirac':
        ans = np.zeros(num_evals)
        ans[0] = 0.5
        ans[num_evals - 1] = 0.5

    elif T_scenario == 'convex_extreme':
        # convex_extreme
        ans = np.array([5, 3, 1, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1,
                        0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 1, 3, 5])
    
    elif T_scenario == 'convex_moderate':
        # convex_moderate
        ans = np.array([3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3])

    else:
        print("unkown dsiruption scenario!")
        exit(1)

    ans = ans / sum(ans)

    return ans


def disruption_duration_interval_prob(T_prob_mass, eval_start_time, eval_end_time, t1, t2):
    # the probability of T between t1 and t2
    ans = 0

    for e in range(len(T_prob_mass)):
        if eval_start_time[e] >= t1 and eval_end_time[e] <= t2:
            ans += T_prob_mass[e]

    return ans


def expected_disruption_duration(T_prob_mass, eval_end_time):
    # eval_end_time is needed since the disruption is assumed to end at eval time points
    # for simplicity; the middle point is used for computing
    sum = 0

    for e in range(len(T_prob_mass)):
        sum += eval_end_time[e] * T_prob_mass[e]

    return int(sum)
