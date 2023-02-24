import numpy as np
import gurobipy as gp
import copy
from gurobipy import GRB

# this evaluation module is able to handle infeasibility
# once infeasibility happens, 1/20 of demands from all OD pairs will be delayed to next epoch (ep)
# this is coarse approximation, since this module is not agent-based.
def evaluation(model_name, y_eval,
               y_n, num_evals, line_capacity,
               num_segments, segment_line, segment_cost,
               path_segment_incidence, boarding_flag,
               num_ods, od_num_paths, od_path2index, Q_evals,
               T_rd, T_prob_mass, eval_interval, M):

    # ---------------- evaluation subroutine ---------------------\
    # probabilities p are recomputed for each time e
    # the only variable is p since y is fixed; and this is quick.
    print("\n\nevaluation starts... evaluating model: {}".format(model_name))
    demand_transfer_fraction = 1/20
    iteration_num_limit = int(1 / demand_transfer_fraction)
    expected_total_user_cost = 0

    for e in range(num_evals):
        # iterate over disruption ending time;
        # Note: disruption ends at end of e
        tmp_user_cost_accu = 0  # used to accumulate user cost under disruption that last till the end of e
        Q_evals_cp = copy.copy(Q_evals)  # make a copy since modifications may be made to this matrix

        for ep in range(num_evals): # evaluation time e'
            # update y setting
            if ep <= e:
                # if disruption is ongoing, update the fleet size setting
                y = y_eval[ep, :]
            else:
                y = y_n  # otherwise, go back to normal

            # segment capacity
            segment_capacity = np.zeros(num_segments)
            for seg in range(num_segments):
                tmp_line = segment_line[seg]
                segment_capacity[seg] = eval_interval / T_rd[tmp_line] * y[tmp_line] * line_capacity[tmp_line]
                #        capacity during a single eval interval           use y

            # segments boarding cost
            segment_boarding_cost = np.zeros(num_segments)
            for seg in range(num_segments):
                tmp_line = segment_line[seg]
                if y[tmp_line] > 0:
                    segment_boarding_cost[seg] = T_rd[tmp_line] / y[tmp_line] / 2
                    #                                               use y
                else:
                    segment_boarding_cost[seg] = M

            # path cost
            path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
                        + np.dot(boarding_flag, np.transpose(segment_boarding_cost))

            # find p by optimization
            # if infeasible, transfer demand to next epoch, then continue loop;
            # otherwise, break;
            for count in range(iteration_num_limit):
                # optimize path flow choice under capacity constraints
                global m_eval
                m_eval = gp.Model("m_eval")
                m_eval.Params.LogToConsole = 0

                # add variables
                # only need to add p
                # syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
                for od in range(num_ods):
                    for path in range(od_num_paths[od]):
                        # note: all path var are added
                        exec("p_{}_{} = m_eval.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())

                # add OD flow conservation constraints
                # syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
                for od in range(num_ods):
                    const_str = ""
                    count = 0
                    for path in range(od_num_paths[od]):
                        if count != 0:
                            const_str += " + "
                        const_str += "p_{}_{}".format(od, path)
                        count += 1
                    const_str += " == 1.0"
                    const_str = "m_eval.addConstr(" + const_str + ", 'od_{}')".format(od)
                    exec(const_str, globals())

                # add capacity constraints
                # syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
                for seg in range(num_segments):
                    const_str = ""
                    count = 0
                    for od in range(num_ods):
                        for path in range(od_num_paths[od]):
                            path_id = od_path2index[od][path]
                            if path_segment_incidence[path_id][seg] == 1:
                                if count != 0:
                                    const_str += " + "
                                const_str += "{:.3f}".format(Q_evals_cp[od, ep])
                                #                    demand during eval interval
                                const_str += " * "
                                const_str += "p_{}_{}".format(od, path)
                                count += 1
                    if const_str == "":
                        continue
                    const_str += " <= {}".format(segment_capacity[seg])
                    const_str = "m_eval.addConstr(" + const_str + ", 'seg_cap_{}')".format(seg)
                    exec(const_str, globals())

                # add objective
                # syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
                #         m.setObjective(obj)
                obj_str = "obj = "
                count = 0
                for od in range(num_ods):
                    for path in range(od_num_paths[od]):
                        path_id = od_path2index[od][path]
                        if count != 0:
                            obj_str += " + "
                        # terms are like: OD_flow * path cost * path_ratio
                        # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
                        tmp_coeff = Q_evals_cp[od, ep] * path_cost[path_id]
                        #      demand in eval interval
                        obj_str += "{:.3f} * p_{}_{}".format(tmp_coeff, od, path)
                        count += 1
                exec(obj_str, globals())
                exec("m_eval.setObjective(obj)", globals())

                m_eval.optimize()

                # accumulate the user cost
                try:
                    tmp = obj.getValue()
                    tmp_user_cost_accu += tmp
                    # if feasible, break out of loop
                    break
                except:  # if infeasible
                    if count == iteration_num_limit:
                        print("iteration limit reached!")
                        exit(1)
                    if ep == (num_evals - 1):
                        print("error! there exist passengers of the last eval epoch that cannot dissipitate!")
                        exit(1)

                    # transfer demands to next eval time
                    demand_transferred = demand_transfer_fraction * Q_evals_cp[:, ep]
                    Q_evals_cp[:, ep] -= demand_transferred
                    Q_evals_cp[:, ep + 1] += demand_transferred
                    print("case: disruption ending time e =", e, "user overflow at evauation time e' =", ep)
                    # use delay must be accounted!
                    tmp_user_cost_accu += demand_transferred.sum() * eval_interval

        # accumulate the user cost weighted by disruption prob mass
        expected_total_user_cost += tmp_user_cost_accu * T_prob_mass[e]
        m_eval.reset()

    # print results
    return expected_total_user_cost
    # ---------------- end of evaluation subroutine ---------------/
