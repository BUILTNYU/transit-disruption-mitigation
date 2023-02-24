# demand generation functions
import numpy as np


def minute_demand(num_ods, T_ub, q_0_vec, q_max_vec, Q_scenario):
    ans = np.zeros((num_ods, T_ub))

    for od in range(num_ods):
        for t in range(T_ub):

            if Q_scenario == 'uniform':
                # uniform (=q_0)
                ans[od, t] = q_0_vec[od]

            elif Q_scenario == 'concave':
                # concave
                tp = t + 1  # t'
                Q_t = -4/3/(T_ub**2)*(q_max_vec[od] - q_0_vec[od])*(t**3) + 2/T_ub*(q_max_vec[od] - q_0_vec[od])*(t**2) + q_0_vec[od]*t
                Q_tp = -4/3/(T_ub**2)*(q_max_vec[od] - q_0_vec[od])*(tp**3) + 2/T_ub*(q_max_vec[od] - q_0_vec[od])*(tp**2) + q_0_vec[od]*tp
                ans[od, t] = Q_tp - Q_t

            elif Q_scenario == 'convex':
                # convex
                tp = t + 1  # t'
                Q_t = 4/3/(T_ub**2)*(q_max_vec[od] - q_0_vec[od])*(t**3) - 2/T_ub*(q_max_vec[od] - q_0_vec[od])*(t**2) + q_max_vec[od]*t
                Q_tp = 4/3/(T_ub**2)*(q_max_vec[od] - q_0_vec[od])*(tp**3) - 2/T_ub*(q_max_vec[od] - q_0_vec[od])*(tp**2) + q_max_vec[od]*tp
                ans[od, t] = Q_tp - Q_t

            elif Q_scenario == 'increasing':
                # monotone_increasing (from q_0 to q_max)
                slope = (q_max_vec[od] - q_0_vec[od]) / T_ub
                ans[od, t] = q_0_vec[od] + slope * t

            elif Q_scenario == 'decreasing':
                # monotone_decreasing (from q_max to q_0)
                slope = (q_0_vec[od] - q_max_vec[od]) / T_ub
                ans[od, t] = q_max_vec[od] + slope * t
            else:
                print("unknown demand scenario!")
                exit(1)

    return ans