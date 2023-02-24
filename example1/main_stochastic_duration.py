# stochastic duration case
# usage:
#                             arg1        arg2  arg3  arg4       arg5       arg6
# main_stochastic_duration.py Q_scenario  q_0  q_max  T_scenario num_stages  alpha
# 
# default values:
# Q_scenario: 'concave'
# q_0: 10
# q_max: 15
# T_scenario: 'decreasing'
# num_stages: 6
# alpha: 10

# 5 models are tested:
# 'lla': local line adjustment
# 'bb': bus bridging
# 'bm': basic model
# 'msm': multi-stage model
# 'itm': initialization time model

# to set x, y variable to be integers (Groubi):
# m.addVar(vType=GRB.INTEGER, ...)

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import math
import copy
import generate_cost
import generate_demand
import generate_disruption_duration
import generate_segment
import generate_path
import evaluation

print("stochastic model starts...")
print("setting up...")

# -----------------------  parameters -------------------------------
# get command line args:
#                             arg1       arg2 arg3  arg4       arg5       arg6
# main_stochastic_duration.py Q_scenario q_0 q_max T_scenario num_stages alpha
if len(sys.argv) != 7:
    print("usage: main_stochastic_duration.py Q_scenario q_0 q_max T_scenario num_stages alpha")
    exit(1)
else:
    Q_scenario = sys.argv[1]
    q_0 = float(sys.argv[2])
    q_max = float(sys.argv[3])
    T_scenario = sys.argv[4]
    num_stages = int(sys.argv[5])
    alpha = float(sys.argv[6])

# general settings
#alpha = 10.0  # weight of operator cost vs user cost = 1 / user value of time (VOT) per minute
# T = 60 # disruption duration now is a random variable, given by prob mass
T_ub = 240  # upper bound for disruption duration (minutes); default = 240
M = 100000000  # big-M
MAX_INITIALIZATION_TIME = 6  # default = int(num_evals * 2 / 3)
GUROBI_TIME_LIMIT_BM = 600 * 60  # gurobi time limit for basic model (seconds)
GUROBI_TIME_LIMIT_MSM = 10 * 60  # gurobi time limit for mutistage model (seconds)
GUROBI_TIME_LIMIT_ITM = 10 * 60  # gurobi time limit for mutistage model (seconds)
PRINT_THRESHOLD = 0.001  # threshold for var result printing
RECURSION_LIMIT = 4 * sys.getrecursionlimit()  # used for exec
Y_LOWER_BOUND = 0.1
X_UB = 100
SEGBC_LB = 0.001
SEGBC_UB = 1000
PATHC_LB = 0.001
PATHC_UB = 4000
VOT = 0.1  # value of time; used for printing at last
sys.setrecursionlimit(RECURSION_LIMIT)

# evaluation settings
eval_interval = 10  # time unit for evaluation (minute); default = 10
num_evals = T_ub // eval_interval  # number of eval intervals; default = 24
T_eval_ub = num_evals  # the upperbound of T in terms of number of evaluation intervals
eval_start_time = []  # start time of eval intervals
eval_end_time = []  # end time of eval intervals
for e in range(num_evals):
    eval_start_time.append(eval_interval * e)
    eval_end_time.append(eval_interval * (e + 1))
    # Note: the last time point is not included;
    # use eval_start_time[e] : eval_end_time[e] to do summations

# multistage model settings
#num_stages = 6  # num of stages; default = 6
stage_interval = T_ub // num_stages  # stage interval length (min)  # default = 40
stage_start_time = []  # start time of stages
stage_end_time = []  # end time of stages
for k in range(num_stages):
    stage_start_time.append(stage_interval * k)
    stage_end_time.append(stage_interval * (k + 1))
    # Note: the last time pt not included

stage_start_eval_time = []
# stage starting time in terms of eval index, used for the calculation of probabilities,
# since prob mass is described in terms of eval indices
stage_end_eval_time = []
tmp = stage_interval // eval_interval  # 1 stage = tmp evals
for k in range(num_stages):
    stage_start_eval_time.append(tmp * k)
    stage_end_eval_time.append(tmp * (k + 1))
    # Note: the last time pt not included


# line settings
num_lines = 8
# round trip time of lines
#       0   1   2   3   4   5  6   7
#       L1  L2  L3  L4  L5  L6 L7  L8
T_rd = [36, 36, 96, 96, 20, 8, 36, 16]
# beta = 1.0  # capacity constraints coefficients
capacity_metro = 800  # capacity of each train
capacity_bus = 200  # capacity of each bus

# initial fleet
# Note: dim(y) = num_lines + 1 to include the backup "line"
#       0  1   2   3  4  5  6  7  8
#       L1 L2 L3 L4  L5 L6 L7  L8 backup_bus
y_0s = [3, 3, 16, 16, 0, 0, 0, 0, 4]  # 's' means 'start'
num_metro_vehs = y_0s[0] + y_0s[1]
num_bus_vehs = y_0s[2] + y_0s[3] + y_0s[8]  # backup bus included

# immediately after disruption, y_0'
#       0  1   2   3  4  5  6  7  8
#       L1 L2 L3  L4  L5 L6 L7 L8 backup_bus
y_0p = [0, 3, 16, 16, 2, 1, 0, 0, 4]  # L1 is disrupted, no longer available

# the upper bound of y is:
#               0  1   2   3  4  5  6  7  8
#               L1 L2 L3 L4  L5 L6 L7 L8 backup_bus
y_ub_disrupt = [0, 6, 20, 20, 4, 2, 5, 3, 4]
# determined by the min headway
# disrupted line has UB = 0
# Note: when recovers, the fleet size go back to normal; no need to give the UB

# line capacity generation
#                      L1              L2             L3           L4
#                       L5               L6              L7             L8
line_capacity = [capacity_metro, capacity_metro, capacity_bus, capacity_bus,
                 capacity_metro, capacity_metro, capacity_metro, capacity_bus]

# segment generation
# num_segments: number of segments
# segment_cost: segment cost array
# segment_line: segment_line[i] is the line of segment i
num_segments, segment_cost, segment_line = generate_segment.segment_generation_expanded()

# path enumeration
# num_paths: total number of paths
# path_segment_incidence: path_segment[path_id, seg_id] = 1 if seg_id is on path_id
# boarding_flag: boarding_flag[path_id, seg_id] = 1 if the seg_id on path_id is a boarding link
# od_num_paths: od_num_paths[i] is num of paths of od i
# od_path2index: od_path2index[od, path] gives the path_id of the path that can be indexed by (od, path)
# path_avail_lla: path availability under LLA model
# path_avail_bb: ... BB model
# path_avail_bm: ... BM, MSM and ITM
# path_avail_nm: path availability under normal case (after recovery)
num_paths, path_segment_incidence, boarding_flag, \
    od_num_paths, od_path2index, \
    path_avail_lla, path_avail_bb, path_avail_bm, path_avail_nm = generate_path.path_enumeration_expanded(num_segments)

# operator cost settings
BLT = 100  # bus line transfer cost
BBT = 300  # bus back-up transfer cost
MLT = 200  # metro line transfer cost
MST = 0  # metro short-turn cost
# operator cost matrix
# (num_lines + 1) x (num_lines + 1) matrix, since backup is included
# backup is the 9-th "line" (index 8)
cost = generate_cost.cost_generation(M, BLT, BBT, MLT, MST)


# disruption settings
# 'T' is for disruption duration
# disruption duration settings
# T = 0 means end at the end of eval interval 0;
T_prob_mass = generate_disruption_duration.prob_mass(num_evals, T_scenario)
T_expected_conditioning_eval = [0] * num_evals
# expected disruption duration conditioning on the event
# that the disruption has not ended at eval time e
T_le_eval_prob = np.zeros(num_evals)  # prob of T >= z
# Note: eval time step is used
for e in range(num_evals):
    T_le_eval_prob[e] = T_prob_mass[e: num_evals].sum()

for e in range(num_evals):
    tmp_prob_sum = 0
    tmp_sum = 0
    for ep in range(e, num_evals):
        tmp_sum += eval_end_time[ep] * T_prob_mass[ep]
        #          use eval end time
    tmp_T_expected = int(tmp_sum / T_prob_mass[e:num_evals].sum())
    T_expected_conditioning_eval[e] = math.ceil(tmp_T_expected / eval_interval) - 1
    # Note: at time e, the disruption is expected to end at the end of eval time T_expected_conditioning_eval[e]

# expected disruption duration (int)
T_expected = generate_disruption_duration.expected_disruption_duration(T_prob_mass, eval_end_time)
# disruption is expected to end at the end of minute time T_expected - 1

# prob that disruption ended during stage k
T_stage_prob = np.zeros(num_stages)
# probability that disruption last >= stage k
T_le_stage_prob = np.zeros(num_stages)
for k in range(num_stages):
    T_stage_prob[k] = T_prob_mass[stage_start_eval_time[k]:stage_end_eval_time[k]].sum()
    T_le_stage_prob[k] = T_prob_mass[stage_start_eval_time[k]:num_evals].sum()


# demand settings
# 'q' is for demand density related
# 'Q' is for demand related
# demand parameters
num_ods = 8
q_0_vec = q_0 * np.ones(num_ods)  # demand per minute at the begining
# 12 is good
q_max_vec = q_max * np.ones(num_ods)  # peak demand per minute during the horizon


# demand for each minute in [0, T_ub)
Q_minute = generate_demand.minute_demand(num_ods, T_ub, q_0_vec, q_max_vec, Q_scenario)
# demand in [0, T_exp) for each OD
Q_disrupt_expected = np.zeros(num_ods)
# demand in [expected_T, T_ub] for each OD
Q_undisrupt_expected = np.zeros(num_ods)
for od in range(num_ods):
    Q_disrupt_expected[od] = Q_minute[od, 0:T_expected].sum()
    Q_undisrupt_expected[od] = Q_minute[od, T_expected:T_ub].sum()

# stage demands for each OD
Q_stages = np.zeros((num_ods, num_stages))
# Q_stages_left_include[k]: demand from stage k to K-1 with stage k included;
Q_stages_left_include = np.zeros((num_ods, num_stages))
for od in range(num_ods):
    for k in range(num_stages):
        Q_stages[od, k] = Q_minute[od, stage_start_time[k]:stage_end_time[k]].sum()
for od in range(num_ods):
    for k in range(num_stages):
        Q_stages_left_include[od, k] = Q_minute[od, stage_start_time[k]:T_ub].sum()

# evaluation interval demands for each OD
Q_evals = np.zeros((num_ods, num_evals))
Q_evals_accu_exclude = np.zeros((num_ods, num_evals))  # accumulated demand < end time of eval interval k
Q_evals_accu_include = np.zeros((num_ods, num_evals))  # accumulated demand <= end time of eval interval k
Q_evals_between_include = np.zeros((num_ods, num_evals, num_evals))  # demand in [z, e], e included
Q_evals_left_exclude = np.zeros((num_ods, num_evals))  # demand left > end of eval interval k
Q_evals_left_include = np.zeros((num_ods, num_evals))  # demand left >= end of eval interval k
for od in range(num_ods):
    for e in range(num_evals):
        Q_evals[od, e] = Q_minute[od, eval_start_time[e]:eval_end_time[e]].sum()
for od in range(num_ods):
    for z in range(num_evals):
        Q_evals_accu_exclude[od, z] = Q_evals[od, 0 : z].sum()
        #                              Note: z NOT included, [0, z)
        Q_evals_accu_include[od, z] = Q_evals[od, 0 : z + 1].sum()
for od in range(num_ods):
    for z in range(num_evals): # initialization time of service
        for e in range(z, num_evals): # end time of disruption
            Q_evals_between_include[od, z, e] = Q_evals[od, z: e + 1].sum()
            #                                      [z, e], e included
for od in range(num_ods):
    for e in range(num_evals):
        Q_evals_left_exclude[od, e] = Q_evals[od, e + 1 : num_evals].sum()
        #                                 e not included
        Q_evals_left_include[od, e] = Q_evals[od, e : num_evals].sum()
        #                                 e included


# print settings
print("\n\nsettings summary:")
print("Q_scenario:", Q_scenario)
print("q_0:", q_0)
print("q_max:", q_max)
print("T_scenario:", T_scenario)
print("num_stages:",num_stages)
print("alpha:", alpha)
print("T_ub:", T_ub)
print("MAX_INITIALIZATION_TIME:", MAX_INITIALIZATION_TIME)
print("GUROBI_TIME_LIMIT_BM:", GUROBI_TIME_LIMIT_BM)
print("GUROBI_TIME_LIMIT_MSM:",GUROBI_TIME_LIMIT_MSM)
print("GUROBI_TIME_LIMIT_ITM:",GUROBI_TIME_LIMIT_ITM)
print("eval_interval:",eval_interval)
print("num_lines:",num_lines)
print("capacity_metro:",capacity_metro)
print("capacity_bus:",capacity_bus)
print("BLT:",BLT)
print("BBT:",BBT)
print("MLT:",MLT)
print("MST:",MST)
print("T_prob_mass:",T_prob_mass)
print("num_ods:",num_ods)

# print statistics
print("\n\nstatistics summary:")
# avg headway of metro line = 36 / 3 = 12 min
stat_avg_headway_metro = T_rd[0] / y_0s[0]
print("avg headway of metro line", stat_avg_headway_metro)
# hourly cap of metro line = 60 / 12 * 800 = 4000 psg
stat_hourly_cap_metro = 60 / stat_avg_headway_metro * capacity_metro
print("hourly cap of metro line", stat_hourly_cap_metro)
# avg headway of bus line = 96 / 16 = 6 min
stat_avg_headway_bus = T_rd[2] / y_0s[2]
print("avg headway of bus line", stat_avg_headway_bus)
# hourly cap of bus line = 60 / 6 * 100 = 1000
stat_hourly_cap_bus = 60 / stat_avg_headway_bus * capacity_bus
print("hourly cap of bus line", stat_hourly_cap_bus)
# hourly demand of one OD pair = 10 * 60 = 600
stat_hourly_demand_per_OD = q_0 * 60
print("hourly demand of one OD pair", stat_hourly_demand_per_OD)
















# -------------- local level adjust model (LLA)--------------
# just compute the user cost
# operator cost is zero since back-up bus is not used and no fleet rellocation
print("\n\n\n\n\n\n\n\nlla model starts...")

print("\npart 1:")
# part 1 is about the cost in [0, T_expected)
# the fleet size is just y_0p
y = y_0p

# disruption duration is the expected one
T = T_expected
# Note: expect disruption to last from to [0, T_expected)
# len = T_expected

# ----------- repeated code start----------------\
# segment capacity
segment_capacity = np.zeros(num_segments)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    segment_capacity[seg] = T / T_rd[tmp_line] * y[tmp_line] * line_capacity[tmp_line]
    #                   use T

# segments boarding cost
segment_boarding_cost = np.zeros(num_segments)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if y[tmp_line] > 0:
        segment_boarding_cost[seg] = T_rd[tmp_line] / y[tmp_line] / 2
        # avg_headway / 2
    else:
        segment_boarding_cost[seg] = M

# path cost
path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
            + np.dot(boarding_flag, np.transpose(segment_boarding_cost))
#           path-segment * seg_cost' + boarding_flag * seg_boarding_cost'

# optimize path flow choice under capacity constraints
m = gp.Model("m")

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
            #       use lla
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
            #     use lla
            if count != 0:
                const_str += " + "
            const_str += "p_{}_{}".format(od, path)
            count += 1
    const_str += " == 1.0"
    const_str = "m.addConstr(" + const_str + ", 'od_{}')".format(od)
    exec(const_str, globals())

# add capacity constraints
# syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
for seg in range(num_segments):
    const_str = ""
    count = 0
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            path_id = od_path2index[od][path]
            if path_avail_lla[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:
                #       use lla
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q_disrupt_expected[od])
                #                   this is demand in [0, T_expected)
                const_str += " * "
                const_str += "p_{}_{}".format(od, path)
                count += 1
    if const_str == "":
        continue
    const_str += " <= {}".format(segment_capacity[seg])
    const_str = "m.addConstr(" + const_str + ", 'seg_cap_{}')".format(seg)
    exec(const_str, globals())

# add objective
# syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
#         m.setObjective(obj)
obj = 0
obj_str = "obj = "
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
            #      use lla
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
            path_id = od_path2index[od][path]
            tmp_coeff = Q_disrupt_expected[od] * path_cost[path_id]
            obj_str += "{:.3f} * p_{}_{}".format(tmp_coeff, od, path)
            count += 1
exec(obj_str, globals())
exec("m.setObjective(obj)", globals())

m.optimize()
# ----------- repeated code end----------------/

# store results
for v in m.getVars():
    if v.X >= PRINT_THRESHOLD:
        print('{}: {:.3f}'.format(v.VarName, v.X))

# use cost
user_cost_lla_part1 = obj.getValue()
print('lla model estimated user cost (obj) part 1: {:.3f}'.format(user_cost_lla_part1))


print("\npart 2:")
# part 1 is about the cost in [T_expected, T_ub)
# during this time interval, the disruption is assumed to have recovered
# changes to be made:
# y, T, Q, path_avail

# the fleet size is just y_0
y = y_0s

# disruption duration is the expected one
T = T_ub - T_expected
# horizon [T_expected, T_ub)
# len = T_ub - T_expected

# ----------- repeated code start----------------\
# segment capacity
segment_capacity = np.zeros(num_segments)
for i in range(num_segments):
    tmp_line = segment_line[i]
    segment_capacity[i] = T / T_rd[tmp_line] * y[tmp_line] * line_capacity[tmp_line]
    #                   use T

# segments boarding cost
segment_boarding_cost = np.zeros(num_segments)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if y[tmp_line] > 0:
        segment_boarding_cost[seg] = T_rd[tmp_line] / y[tmp_line] / 2
        # half of avg. headway
    else:
        segment_boarding_cost[seg] = M

# path cost
path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
            + np.dot(boarding_flag, np.transpose(segment_boarding_cost))

# optimize path flow choice under capacity constraints
m.reset()
m = gp.Model("m")

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_nm[od][path] == 1:  # change
            #        use 'nm'
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())
        count += 1

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_nm[od][path] == 1:  # change
            #       use 'nm'
            if count != 0:
                const_str += " + "
            const_str += "p_{}_{}".format(od, path)
            count += 1
    const_str += " == 1.0"
    const_str = "m.addConstr(" + const_str + ", 'od_{}')".format(od)
    exec(const_str, globals())

# add capacity constraints
# syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
for seg in range(num_segments):
    const_str = ""
    count = 0
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            path_id = od_path2index[od][path]
            if path_avail_nm[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:
                #     use 'nm'
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q_undisrupt_expected[od])
                #                 use undisrupted expected demand in time [T_expected, T_ub)
                const_str += " * "
                const_str += "p_{}_{}".format(od, path)
                count += 1
    if const_str == "":
        continue
    const_str += " <= {}".format(segment_capacity[seg])
    const_str = "m.addConstr(" + const_str + ", 'seg_cap_{}')".format(seg)
    exec(const_str, globals())

# add objective
# syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
#         m.setObjective(obj)
obj_str = "obj = "
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_nm[od][path] == 1:  # change
            #        use nm
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
            path_id = od_path2index[od][path]
            tmp_coeff = Q_undisrupt_expected[od] * path_cost[path_id]
            #             use undisrupted expected demand
            obj_str += "{:.3f} * p_{}_{}".format(tmp_coeff, od, path)
            count += 1
exec(obj_str, globals())
exec("m.setObjective(obj)", globals())

m.optimize()
# ----------- repeated code end----------------/

# store results
for v in m.getVars():
    if v.X >= PRINT_THRESHOLD:
        print('{}: {:.3f}'.format(v.VarName, v.X))

# use cost
user_cost_lla_part2 = obj.getValue()
print('lla model estimated user cost (obj) part 2: {:.3f}'.format(user_cost_lla_part2))


# sum up
print("\nsummarizing two parts:")
user_cost_lla = user_cost_lla_part1 + user_cost_lla_part2
op_cost_lla = 0
total_cost_lla = user_cost_lla + op_cost_lla
print('lla model estimated user cost (obj): {:.3f}'.format(user_cost_lla))
print('lla model estimated total cost (obj): {:.3f}'.format(total_cost_lla))

y_lla = y_0p

# prepare evaluation input: y[e, l]
# this matrix stores the actions of operators assuming that disruption is ongoing
# RMK: there is no need to accumulate operator cost
#      since operator cost is the same with model output.
y_eval = np.zeros((num_evals, num_lines + 1))
for e in range(num_evals):
    for l in range(num_lines + 1):
        y_eval[e, l] = y_lla[l]
# y_n is the configuration of fleets when disruption is over
y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends
model_name = 'lla'

# call evaluation function
user_cost_lla_eval =  evaluation.evaluation(model_name, y_eval,
                      y_n, num_evals, line_capacity,
                      num_segments, segment_line, segment_cost,
                      path_segment_incidence, boarding_flag,
                      num_ods, od_num_paths, od_path2index, Q_evals,
                      T_rd, T_prob_mass, eval_interval, M)
total_cost_lla_eval = user_cost_lla_eval + op_cost_lla
print("lla model evaluated total user cost:", user_cost_lla_eval)
print('lla model eevaluated total cost: {:.3f}'.format(total_cost_lla_eval))

















# -------------- bus bridging model (BB) ---------------------
print("\n\n\n\n\n\n\n\nbb model starts...")
print("\npart 1:")
# part 1 is about the cost in [0, T_expected]
T = T_expected

# vars over iterations
optimal_num_bb_bus = M
user_cost_bb_part1 = M
op_cost_bb_part1 = M
total_cost_bb_part1 = M
y_bb = [0] * (num_lines + 1)

for num_bb_bus in range(0, y_0p[8] + 1):
    # iterate over possible settings of number of back up bus used
    # Note: under bb model, we allow no bus to bridge the broken link
    print("\n\ncurrent number of buses used for bridging:", num_bb_bus)

    # update fleet
    # make a deepcopy since modifications will be made
    y = copy.deepcopy(y_0p)
    #    L1...        L5 L6 L7 L8 backup_bus
    y[7] += num_bb_bus
    y[8] -= num_bb_bus

    # ----------- repeated code start----------------\
    # segment capacity
    segment_capacity = np.zeros(num_segments)
    for seg in range(num_segments):
        tmp_line = segment_line[seg]
        segment_capacity[seg] = T / T_rd[tmp_line] * y[tmp_line] * line_capacity[tmp_line]
        #                   use T

    # segments boarding cost
    segment_boarding_cost = np.zeros(num_segments)
    for seg in range(num_segments):
        if y[segment_line[seg]] > 0:
            segment_boarding_cost[seg] = T_rd[segment_line[seg]] / y[segment_line[seg]] / 2
            # half of the avg. headway
        else:
            segment_boarding_cost[seg] = M

    # path cost
    path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
                + np.dot(boarding_flag, np.transpose(segment_boarding_cost))
    #           path-segment * seg_cost' + boarding_flag * seg_boarding_cost'

    # optimize path flow choice under capacity constraints
    m.reset()
    m = gp.Model("m")

    # add variables
    # syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bb[od][path] == 1:  # change!!
                #       use bb
                exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())

    # add OD flow conservation constraints
    # syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
    for od in range(num_ods):
        const_str = ""
        count = 0
        for path in range(od_num_paths[od]):
            if path_avail_bb[od][path] == 1:  # change!!
                #       use bb
                if count != 0:
                    const_str += " + "
                const_str += "p_{}_{}".format(od, path)
                count += 1
        const_str += " == 1.0"
        const_str = "m.addConstr(" + const_str + ", 'od_{}')".format(od)
        exec(const_str, globals())

    # add capacity constraints
    # syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
    for seg in range(num_segments):
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                path_id = od_path2index[od][path]
                if path_avail_bb[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:
                    #      use bb
                    if count != 0:
                        const_str += " + "
                    const_str += "{:.3f}".format(Q_disrupt_expected[od])
                    #                         use disrupt expected
                    const_str += " * "
                    const_str += "p_{}_{}".format(od, path)
                    count += 1
        if const_str == "":
            continue
        const_str += " <= {}".format(segment_capacity[seg])
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_{}')".format(seg)
        exec(const_str, globals())

    # add objective
    # syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
    #         m.setObjective(obj)
    obj_str = "obj = "
    count = 0
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bb[od][path] == 1:
                #        use bb
                if count != 0:
                    obj_str += " + "
                # terms are like: OD_flow * path cost * path_ratio
                # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
                path_id = od_path2index[od][path]
                tmp_coeff = Q_disrupt_expected[od] * path_cost[path_id]
                obj_str += "{:.3f} * p_{}_{}".format(tmp_coeff, od, path)
                count += 1
    exec(obj_str, globals())
    exec("m.setObjective(obj)", globals())

    m.Params.LogToConsole = 0
    m.optimize()
    # ----------- repeated code end----------------/

    tmp_user_cost = obj.getValue()
    tmp_op_cost = 2 * alpha * num_bb_bus * cost[8, 7] # relocate from backup to L8
    tmp_total_cost = tmp_user_cost + tmp_op_cost
    if tmp_total_cost < total_cost_bb_part1:
        print("optimal num_bb_bus and costs updated to:")
        # store results
        for v in m.getVars():
            if v.X >= PRINT_THRESHOLD:
                print('{}: {:.3f}'.format(v.VarName, v.X))

        optimal_num_bb_bus = num_bb_bus
        user_cost_bb_part1 = tmp_user_cost
        op_cost_bb_part1 = tmp_op_cost
        total_cost_bb_part1 = tmp_total_cost
        y_bb = y
    
print("optimal number of bus used:", optimal_num_bb_bus)
print('bb model estimated user cost (obj) part 1: {:.3f}'.format(user_cost_bb_part1))
print('bb model estimated operator cost part 1: {:.3f}'.format(op_cost_bb_part1))
print('bb model estimated total cost part 1: {:.3f}'.format(total_cost_bb_part1))


print("\npart 2:")
# part 2 cost is the same with LLA part 2!
print("the same with LLA part 2")
user_cost_bb_part2 = user_cost_lla_part2
op_cost_bb_part2 = 0
total_cost_bb_part2 = user_cost_bb_part2 + op_cost_bb_part2
print('bb model estimated user cost (obj) part 2: {:.3f}'.format(user_cost_bb_part2))
print('bb model estimated operator cost part 2: {:.3f}'.format(op_cost_bb_part2))
print('bb model estimated total cost part 2: {:.3f}'.format(total_cost_bb_part2))

print("\nsummarizing two parts:")
user_cost_bb = user_cost_bb_part1 + user_cost_bb_part2
op_cost_bb = op_cost_bb_part1 + op_cost_bb_part2
total_cost_bb = total_cost_bb_part1 + total_cost_bb_part2
print("optimal number of bus used:", optimal_num_bb_bus)
print('bb model estimated user cost (obj): {:.3f}'.format(user_cost_bb))
print('bb model estimated operator cost: {:.3f}'.format(op_cost_bb))
print('bb model estimated total cost: {:.3f}'.format(total_cost_bb))


# prepare evaluation input: y[e, l]
# this matrix stores the actions of operators assuming that disruption is ongoing
# RMK: there is no need to accumulate operator cost
#      since operator cost is the same with model output.
y_eval = np.zeros((num_evals, num_lines + 1))
for e in range(num_evals):
    for l in range(num_lines + 1):
        y_eval[e, l] = y_bb[l]
# y_n is the configuration of fleets when disruption is over
y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends
model_name = 'bb'

# call evaluation function
user_cost_bb_eval = evaluation.evaluation(model_name, y_eval,
                      y_n, num_evals, line_capacity,
                      num_segments, segment_line, segment_cost,
                      path_segment_incidence, boarding_flag,
                      num_ods, od_num_paths, od_path2index, Q_evals,
                      T_rd, T_prob_mass, eval_interval, M)
total_cost_bb_eval = user_cost_bb_eval + op_cost_bb
print("bb model evaluated total user cost:", user_cost_bb_eval)
print('bb model eevaluated total cost: {:.3f}'.format(total_cost_bb_eval))



















# -------------- basic model (BM)------------------------------
# use expected disruption duration to generate y
print("\n\n\n\n\n\n\n\nbm model starts...")
# vars are:
# p - path choice, as before
# y - fleet change
# x - relocation decisions

print("\npart 1:")
# part 1 is about the cost in [0, T_expected]

# for test only
#T = 60
#for i in range(num_ods):
#    Q_disrupt_expected[i] = -4/3/(T_ub**2)*(q_max[i] - q_0[i])*(T**3) + 2/T_ub*(q_max[i] - q_0[i])*(T**2) + q_0[i]*T

# T_expected will be used

m.reset()
# set model to print
m = gp.Model("m")
m.Params.LogToConsole = 1
m.setParam('TimeLimit', GUROBI_TIME_LIMIT_BM)

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
# add p
# Note that every path is possible under basic model
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bm[od][path] == 1:
            #       use 'bm'
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())

# add y
for line in range(num_lines + 1):
    if line == 0:
        exec("y_{} = m.addVar(lb=0.0, ub={}, name='y_{}')".format(line, y_ub_disrupt[line], line), globals())
    else:
        exec("y_{} = m.addVar(lb={}, ub={}, name='y_{}')".format(line,Y_LOWER_BOUND, y_ub_disrupt[line], line), globals())
    #                                                               use y_ub

# add x
for l in range(num_lines + 1):
    for lp in range(num_lines + 1):
        if l == 0 or lp == 0:
            exec("x_{}_{} = m.addVar(lb=0.0, ub=0.0, name='x_{}_{}')".format(l, lp, l, lp), globals())
        else:
            exec("x_{}_{} = m.addVar(lb=0.0, ub={}, name='x_{}_{}')".format(l, lp, X_UB, l, lp), globals())

# add segbc var (segment boarding cost) for temporary use
# only LB is specified, although we can specify a large UB
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if y_ub_disrupt[tmp_line] > 0:
        exec("segbc_{} = m.addVar(lb={}, ub={}, name='segbc_{}')".format(seg, SEGBC_LB, SEGBC_UB, seg))
    else:
        # segc will be M
        exec("segbc_{} = m.addVar(lb={}, name='segbc_{}')".format(seg, SEGBC_LB, seg))

# add pathc var (path cost) for temporary use
# only LB is specified, although we can specify a large UB
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bm[od][path] == 1:
            #        use bm
            path_id = od_path2index[od][path]
            exec("pathc_{} = m.addVar(lb={}, ub={}, name='pathc_{}')".format(path_id, PATHC_LB, PATHC_UB, path_id))

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_bm[od][path] == 1:
            #        use bm
            if count != 0:
                const_str += " + "
            const_str += "p_{}_{}".format(od, path)
            count += 1
    const_str += " == 1.0"
    const_str = "m.addConstr(" + const_str + ", 'od_{}')".format(od)
    exec(const_str, globals())

# add capacity constraints
# syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
for seg in range(num_segments):
    const_str = ""
    count = 0
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            path_id = od_path2index[od][path]
            if path_avail_bm[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:
                #         use bm
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q_disrupt_expected[od])
                #                           use disrupt expected - demand in time [0, T_expected)
                const_str += " * "
                const_str += "p_{}_{}".format(od, path)
                count += 1
    if const_str == "":
        continue
    const_str += " - "
    tmp_line = segment_line[seg]
    tmp_coeff = T_expected / T_rd[tmp_line] * line_capacity[tmp_line]
    #       use T_expected - expected disruption interval [0, T_expected)
    const_str += "{:.3f} * y_{}".format(tmp_coeff, tmp_line)
    const_str += " <= 0"
    const_str = "m.addConstr(" + const_str + ", 'seg_cap_{}')".format(seg)
    exec(const_str, globals())

# add constraints on y
# metro fleet
const_str = "y_1 + y_4 + y_5 + y_6 == {}".format(num_metro_vehs)
const_str = "m.addConstr(" + const_str + ", 'y_metro')"
exec(const_str, globals())
# bus fleet
const_str = "y_2 + y_3 + y_7 + y_8 == {}".format(num_bus_vehs)
const_str = "m.addConstr(" + const_str + ", 'y_bus')"
exec(const_str, globals())
# L7 and L2 will interfere; hence we add a constraint on y[1] + y[6] <= L2 capacity
const_str = "y_1 + y_6 <= {}".format(y_ub_disrupt[1])
const_str = "m.addConstr(" + const_str + ", 'y_metro_overlap')"
exec(const_str, globals())

# add constraints on x
for l in range(num_lines + 1):
    const_str = ""
    count = 0
    for lp in range(num_lines + 1):
        if count != 0:
            const_str += " + "
        const_str += "x_{}_{}".format(l, lp)
        count += 1
    for lp in range(num_lines + 1):
        const_str += " - x_{}_{}".format(lp, l)
    const_str += " + y_{}".format(l)
    const_str += " == {}".format(y_0p[l])
    const_str = "m.addConstr(" + const_str + ", 'x_conservation_{}')".format(l)
    exec(const_str, globals())
for l in range(num_lines + 1):
    const_str = "x_{}_{} == 0".format(l, l)
    const_str = "m.addConstr(" + const_str + ", 'x_ll=0_{}')".format(l)
    exec(const_str, globals())

# compute segbc (segment boarding cost)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if (y_ub_disrupt[tmp_line] > 0):  # if you don't add this condition, the problem would be infeasible!
        const_str = "segbc_{} * y_{}".format(seg, tmp_line)
        const_str += " >= {:.3f}".format(T_rd[tmp_line] / 2)
        # Note: this is nonlinear constraint! (obj still non-linear)
    else:
        const_str = "segbc_{}".format(seg)
        const_str += " == {:.3f}".format(M)
    const_str = "m.addConstr(" + const_str + ", 'segbc_{}')".format(seg)
    exec(const_str, globals())

# compute pathc (path cost)
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bm[od][path] == 1:
            #       use bm
            path_id = od_path2index[od][path]
            const_str = "pathc_{}".format(path_id)
            tmp_tc = 0  # traveling cost
            for seg in range(num_segments):
                if boarding_flag[path_id, seg] == 1:  # Note: here don't use path_segment_incidence!
                    const_str += " - segbc_{}".format(seg)
                if path_segment_incidence[path_id, seg] == 1:
                    tmp_tc += segment_cost[seg]
            const_str += " == {:.3f}".format(tmp_tc)
            const_str = "m.addConstr(" + const_str + ", 'pathc_')".format(path_id)
            exec(const_str, globals())

# add objective
# syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
#         m.setObjective(obj)
obj_str = "obj = "
count = 0
# user cost terms
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bm[od][path] == 1:
            #        use bm
            path_id = od_path2index[od][path]
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * p_od_path * path_cost[od_path2index[od][path]]
            obj_str += "{:.3f} * p_{}_{} * pathc_{}".format(Q_disrupt_expected[od], od, path, path_id)
            #                                             use disrupt expected
            count += 1

# operator cost:
for l in range(num_lines + 1):
    for lp in range(num_lines + 1):
        tmp_coeff = 2 * alpha * cost[l, lp]
        obj_str += " + {:3f} * x_{}_{}".format(tmp_coeff, l, lp)

exec(obj_str, globals())
exec("m.setObjective(obj)", globals())

m.params.NonConvex = 2
m.optimize()

# store results
total_cost_bm_part1 = obj.getValue()
op_cost_bm_part1 = 0
y_bm = [0] * (num_lines + 1)
for v in m.getVars():
    if v.X >= PRINT_THRESHOLD:
        print('{}: {:.3f}'.format(v.VarName, v.X))

    if "x_" in v.VarName:
        # x_{}_{}
        #   stage l' l
        str_temp = v.VarName.split('_')
        l = int(str_temp[1])
        lp = int(str_temp[2])
        op_cost_bm_part1 += 2 * alpha * cost[l, lp] * v.X

    if "y_" in v.VarName:
        # y_{}
        str_temp = v.VarName.split('_')
        l = int(str_temp[1])
        y_bm[l] = v.X

# use cost
user_cost_bm_part1 = total_cost_bm_part1 - op_cost_bm_part1
print('bm model estimated user cost part 1: {:.3f}'.format(user_cost_bm_part1))
print('bm model estimated operator cost part 1: {:.3f}'.format(op_cost_bm_part1))
print('bm model estimated total cost part 1: {:.3f}'.format(total_cost_bm_part1))

print("\npart 2:")
# part 2 cost is the same with LLA part 2!
print("the same with LLA part 2")
user_cost_bm_part2 = user_cost_lla_part2
op_cost_bm_part2 = 0
total_cost_bm_part2 = user_cost_bm_part2 + op_cost_bm_part2
print('bm model estimated user cost (obj) part 2: {:.3f}'.format(user_cost_bm_part2))
print('bm model estimated operator cost part 2: {:.3f}'.format(op_cost_bm_part2))
print('bm model estimated total cost part 2: {:.3f}'.format(total_cost_bm_part2))

print("\nsummarizing two parts:")
user_cost_bm = user_cost_bm_part1 + user_cost_bm_part2
op_cost_bm = op_cost_bm_part1 + op_cost_bm_part2
total_cost_bm = total_cost_bm_part1 + total_cost_bm_part2
print('bm model estimated user cost (obj): {:.3f}'.format(user_cost_bm))
print('bm model estimated operator cost: {:.3f}'.format(op_cost_bm))
print('bm model estimated total cost: {:.3f}'.format(total_cost_bm))


# prepare evaluation input: y[e, l]
# this matrix stores the actions of operators assuming that disruption is ongoing
# RMK: there is no need to accumulate operator cost
#      since operator cost is the same with model output.
y_eval = np.zeros((num_evals, num_lines + 1))
for e in range(num_evals):
    for l in range(num_lines + 1):
        y_eval[e, l] = y_bm[l]
# y_n is the configuration of fleets when disruption is over
y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends
model_name = 'bm'

# call evaluation function
user_cost_bm_eval = evaluation.evaluation(model_name, y_eval,
                      y_n, num_evals, line_capacity,
                      num_segments, segment_line, segment_cost,
                      path_segment_incidence, boarding_flag,
                      num_ods, od_num_paths, od_path2index, Q_evals,
                      T_rd, T_prob_mass, eval_interval, M)

total_cost_bm_eval = user_cost_bm_eval + op_cost_bm
print("bm model evaluated total user cost:", user_cost_bm_eval)
print('bm model evaluated total cost: {:.3f}'.format(total_cost_bm_eval))


















# -------------- multi-stage model (MSM)--------------------------
print("\n\n\n\n\n\n\n\nmulti-stage model (MSM) starts...")
# vars are:
# p_s0,..., p_s(K-1) - path choice during stage k, if disruption continues;
# p_n1,..., p_n(K-1) - path choice from stage k to (K-1), if disruption ends at (k-1);
# Note: disruption last for at least one stage
# y_s0, ..., y_s(K-1) - fleet change at the start of the stage
# x_s0,..., y_s(K-1) - relocation decisions at the start of the stage
#
# where's' means stage;
# 'n' means normal
# 's0',... means stage
# 'K' is num_stages

# fleet size after recovery
y_n = y_0s  # we assume that the fleet size go back to y_0 when disruption ends

m.reset()
m = gp.Model("m")
m.Params.LogToConsole = 1
m.setParam('TimeLimit', GUROBI_TIME_LIMIT_MSM)

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
# Note: add one set of variables for each stage k
# add variables
# add p_s, p_n
# p_s: user choices at stage s
# p_n: user choices at stage >= n when it's normal at stage >=n
for k in range(num_stages):
    # Note that every path is possible under basic model
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use 'bm'
                exec("p_s{}_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_s{}_{}_{}')".format(k, od, path, k, od, path), globals())

            if k > 0 and path_avail_nm[od][path] == 1:
                #         use 'nm
                exec("p_n{}_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_n{}_{}_{}')".format(k, od, path, k, od, path), globals())

# add y_s
for k in range(num_stages):
    for line in range(num_lines + 1):
        if line == 0:
            exec("y_s{}_{} = m.addVar(lb=0.0, ub={}, name='y_s{}_{}')".format(k, line, y_ub_disrupt[line], k, line), globals())
        else:
            exec("y_s{}_{} = m.addVar(lb={}, ub={}, name='y_s{}_{}')".format(k, line, Y_LOWER_BOUND, y_ub_disrupt[line], k, line), globals())

# add x_s
for k in range(num_stages):
    for l in range(num_lines + 1):
        for lp in range(num_lines + 1):
            if l == 0 or lp == 0:
                exec("x_s{}_{}_{} = m.addVar(lb=0.0, ub=0.0, name='x_s{}_{}_{}')".format(k, l, lp, k, l, lp), globals())
            else:
                exec("x_s{}_{}_{} = m.addVar(lb=0.0, name='x_s{}_{}_{}')".format(k, l, lp, k, l, lp), globals())

# add segbc_s var (segment boarding cost) for temporary use
# only LB is specified, although we can specify a large UB
# Note: no need to add var for 'n' - they are constants
for k in range(num_stages):
    for seg in range(num_segments):
        exec("segbc_s{}_{} = m.addVar(lb=0.0, name='segbc_s{}_{}')".format(k, seg, k, seg))

# add pathc_s var (path cost) for temporary use
# only LB is specified, although we can specify a large UB
# Note: no need to add var for 'n' - they are constants
for k in range(num_stages):
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use bm
                path_id = od_path2index[od][path]
                exec("pathc_s{}_{} = m.addVar(lb=0.0, name='pathc_s{}_{}')".format(k, path_id, k, path_id))

# add constraints
# add OD flow conservation constraints for 's' and 'n'
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for k in range(num_stages):
    for od in range(num_ods):
        # 's'
        const_str = ""
        count = 0
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use 'bm'
                if count != 0:
                    const_str += " + "
                const_str += "p_s{}_{}_{}".format(k, od, path)
                count += 1
        const_str += " == 1.0"
        const_str = "m.addConstr(" + const_str + ", 'od_s{}_{}')".format(k, od)
        exec(const_str, globals())

        # 'n'
        if k > 0:
            const_str = ""
            count = 0
            for path in range(od_num_paths[od]):
                if path_avail_nm[od][path] == 1:
                    #        use 'nm'
                    if count != 0:
                        const_str += " + "
                    const_str += "p_n{}_{}_{}".format(k, od, path)
                    count += 1
            const_str += " == 1.0"
            const_str = "m.addConstr(" + const_str + ", 'od_n{}_{}')".format(k, od)
            exec(const_str, globals())

# add capacity constraints for 's' and 'n'
# syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
# 's'
for k in range(num_stages):
    for seg in range(num_segments):
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                if path_avail_bm[od][path] == 1:
                    #        use 'bm'
                    path_id = od_path2index[od][path]
                    if path_segment_incidence[path_id][seg] == 1:
                        if count != 0:
                            const_str += " + "
                        const_str += "{:.3f}".format(Q_stages[od, k])
                        #                            stage k demand only
                        const_str += " * "
                        const_str += "p_s{}_{}_{}".format(k, od, path)
                        count += 1
        if const_str == "":
            continue
        const_str += " - "
        tmp_line = segment_line[seg]
        tmp_coeff = stage_interval / T_rd[tmp_line] * line_capacity[tmp_line]
        #         use stage interval
        const_str += "{:.3f} * y_s{}_{}".format(tmp_coeff, k, tmp_line)
        #                                    stage k seg capacity
        const_str += " <= 0"
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_s{}_{}')".format(k, seg)
        exec(const_str, globals())

# 'n'
for k in range(1, num_stages):
    # start from 1, since disruption last for at least one stage
    for seg in range(num_segments):
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                if path_avail_nm[od][path] == 1:
                    #        use 'nm'
                    path_id = od_path2index[od][path]
                    if path_segment_incidence[path_id][seg] == 1:
                        if count != 0:
                            const_str += " + "
                        const_str += "{:.3f}".format(Q_stages_left_include[od, k])
                        #                            demand left
                        const_str += " * "
                        const_str += "p_n{}_{}_{}".format(k, od, path)
                        count += 1
        if const_str == "":
            continue
        const_str += " - "
        tmp_line = segment_line[seg]
        tmp_coeff = (num_stages - k) * stage_interval * y_n[tmp_line] / T_rd[tmp_line] * line_capacity[tmp_line]
        #         use length of stages left
        const_str += "{:.3f}".format(tmp_coeff)
        #                     always 'n'  change for multistage!!
        const_str += " <= 0"
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_n{}_{}')".format(k, seg)
        exec(const_str, globals())

# add constraints on y_s
for k in range(num_stages):
    # metro fleet
    const_str = "y_s{}_1 + y_s{}_4 + y_s{}_5 + y_s{}_6 == {}".format(k, k, k, k, num_metro_vehs)
    const_str = "m.addConstr(" + const_str + ", 'y_s{}_metro')".format(k)
    exec(const_str, globals())
    # bus fleet
    const_str = "y_s{}_2 + y_s{}_3 + y_s{}_7 + y_s{}_8 == {}".format(k, k, k, k, num_bus_vehs)
    const_str = "m.addConstr(" + const_str + ", 'y_s{}_bus')".format(k)
    exec(const_str, globals())
    # L7 and L2 will interfere; hence we add a constraint on y[1] + y[6] <= L2 capacity
    const_str = "y_s{}_1 + y_s{}_6 <= {}".format(k, k, y_ub_disrupt[1])
    const_str = "m.addConstr(" + const_str + ", 'y_s{}_metro_overlap')".format(k)
    exec(const_str, globals())

if False:
    # for test purpose only
    # test the optimality of Gurobi solution for msm model
    # these fixed values the results from itm model
    # msm with fixed y and x from itm has better result than msm with free y and x!!!
    k = 0
    const_str = "m.addConstr(y_s{}_0 == 0.0, 'y_s{}_0_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_1 == 3.0, 'y_s{}_1_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_2 == 16.0, 'y_s{}_2_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_3 == 15.99999, 'y_s{}_3_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_4 == 1.99999, 'y_s{}_4_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_5 == 1.0, 'y_s{}_5_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_6 == 0.00001, 'y_s{}_6_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_7 == 0.00001, 'y_s{}_7_test')".format(k, k)
    exec(const_str, globals())
    const_str = "m.addConstr(y_s{}_8 == 4.0, 'y_s{}_8_test')".format(k, k)
    exec(const_str, globals())

    for k in range(1, num_stages):
        const_str = "m.addConstr(y_s{}_0 == 0.0, 'y_s{}_0_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_1 == 4.623, 'y_s{}_1_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_2 == 15.428, 'y_s{}_2_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_3 == 14.97599, 'y_s{}_3_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_4 == 0.07499, 'y_s{}_4_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_5 == 1.183, 'y_s{}_5_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_6 == 0.11901, 'y_s{}_6_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_7 == 1.64201, 'y_s{}_7_test')".format(k, k)
        exec(const_str, globals())
        const_str = "m.addConstr(y_s{}_8 == 3.954, 'y_s{}_8_test')".format(k, k)
        exec(const_str, globals())

    x_dict = {}
    x_dict[(0, 3, 7)] = 0.00001
    x_dict[(0, 4, 6)] = 0.00001

    x_dict[(1, 2, 7)] = 0.572
    x_dict[(1, 3, 7)] = 1.024
    x_dict[(1, 4, 1)] = 1.623
    x_dict[(1, 4, 5)] = 0.183
    x_dict[(1, 4, 6)] = 0.119
    x_dict[(1, 8, 7)] = 0.046


    for k in range(num_stages):
        for l in range(num_lines + 1):
            for lp in range(num_lines + 1):
                if (k, l, lp) in x_dict:
                    const_str = "m.addConstr(x_s{}_{}_{} == {}, 'x_s{}_{}_{}_test')".format(k, l, lp, x_dict[(k, l, lp)], k, l, lp)
                    exec(const_str, globals())
                else:
                    const_str = "m.addConstr(x_s{}_{}_{} == 0.0, 'x_s{}_{}_{}_test')".format(k, l, lp, k, l, lp)
                    exec(const_str, globals())

# add constraints on x_s
for k in range(num_stages):
    for l in range(num_lines + 1):
        const_str = ""
        count = 0
        for lp in range(num_lines + 1):
            if count != 0:
                const_str += " + "
            const_str += "x_s{}_{}_{}".format(k, l, lp)
            count += 1
        for lp in range(num_lines + 1):
            const_str += " - x_s{}_{}_{}".format(k, lp, l)
        const_str += " + y_s{}_{}".format(k, l)
        # note the RHS depend on k
        if k == 0:
            const_str += " == {}".format(y_0p[l])
        else:
            const_str += " - y_s{}_{} == 0".format(k - 1, l)
        #                            fleet size from last stage
        const_str = "m.addConstr(" + const_str + ", 'x_conservation_s{}_{}')".format(k, l)
        exec(const_str, globals())

    for l in range(num_lines + 1):
        const_str = "x_s{}_{}_{} == 0".format(k, l, l)
        const_str = "m.addConstr(" + const_str + ", 'x_s{}_ll=0_{}')".format(k, l)
        exec(const_str, globals())

# compute segbc_s (segment boarding cost)
for k in range(num_stages):
    for seg in range(num_segments):
        tmp_line = segment_line[seg]
        if y_ub_disrupt[tmp_line] > 0:  # if you don't add this condition, the problem would be infeasible!
            const_str = "segbc_s{}_{} * y_s{}_{}".format(k, seg, k, tmp_line)
            const_str += " >= {:.3f}".format(T_rd[tmp_line] / 2)
            # Note: this is nonlinear constraint! (obj still non-linear)
        else:
            const_str = "segbc_s{}_{}".format(k, seg)
            const_str += " == {:.3f}".format(M)
        const_str = "m.addConstr(" + const_str + ", 'segbc_s{}_{}')".format(k, seg)
        exec(const_str, globals())

# compute segbc_n (segment boarding cost)  Note: constants
segbc_n = np.zeros(num_segments)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if y_n[tmp_line] > 0:
        segbc_n[seg] = T_rd[tmp_line] / y_n[tmp_line] / 2
        #                             use y_n
    else:
        segbc_n[seg] = M

# compute pathc_s (path cost)
for k in range(num_stages):
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use 'bm'
                path_id = od_path2index[od][path]
                const_str = "pathc_s{}_{}".format(k, path_id)
                tmp_tc = 0 # travel cost accu
                for seg in range(num_segments):
                    if boarding_flag[path_id, seg] == 1:  # Note: here don't use path_segment_incidence!
                        const_str += " - segbc_s{}_{}".format(k, seg)
                    if path_segment_incidence[path_id, seg] == 1:
                        tmp_tc += segment_cost[seg]
                const_str += " == {:.3f}".format(tmp_tc)
                const_str = "m.addConstr(" + const_str + ", 'pathc_s{}_{}')".format(k, path_id)
                exec(const_str, globals())

# compute pathc_n (path cost)  Note: constants
pathc_n = np.zeros(num_paths)
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_nm[od][path] == 1:
            #      use "nm"
            path_id = od_path2index[od][path]
            tmp_ttc = 0  # total cost accu
            # Note: this is different from tmp_tc, which is just accumulation of traveling cost
            for seg in range(num_segments):
                if boarding_flag[path_id, seg] == 1:  # Note: here don't use path_segment_incidence!
                    tmp_ttc += segbc_n[seg]
                if path_segment_incidence[path_id, seg] == 1:
                    tmp_ttc += segment_cost[seg]
            pathc_n[path_id] = tmp_ttc

# add objective
# syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
#         m.setObjective(obj)
obj_str = "obj = "
count = 0
# user cost terms
# part 1)
for k in range(num_stages):
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use bm
                path_id = od_path2index[od][path]
                if count != 0:
                    obj_str += " + "
                # terms are like: OD_flow * path cost * path_ratio
                # Q[od] * p_od_path * path_cost[od_path2index[od][path]]
                tmp_coeff = Q_stages[od, k] * T_le_stage_prob[k]
                obj_str += "{:.3f} * p_s{}_{}_{} * pathc_s{}_{}".format(tmp_coeff, k, od, path, k, path_id)
                count += 1

# part 2)
for k in range(1, num_stages):
    #         Note: start from 1, since disruption last for at least one stage
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_nm[od][path] == 1:
                #         use nm
                path_id = od_path2index[od][path]
                # terms are like: OD_flow * path cost * path_ratio
                # Q[od] * p_od_path * path_cost[od_path2index[od][path]]
                tmp_coeff = Q_stages_left_include[od, k] * T_stage_prob[k - 1] * pathc_n[path_id]
                #                                                      use k-1
                obj_str += " + {:.3f} * p_n{}_{}_{}".format(tmp_coeff, k, od, path)

# part 3) operator cost:
for k in range(num_stages):
    for l in range(num_lines + 1):
        for lp in range(num_lines + 1):
            tmp_coeff = 2 * alpha * cost[l, lp] * T_le_stage_prob[k]
            obj_str += " + {:.3f} * x_s{}_{}_{}".format(tmp_coeff, k, l, lp)

exec(obj_str, globals())
exec("m.setObjective(obj)", globals())

m.params.NonConvex = 2
m.optimize()

# store results
# operator cost
op_cost_msm = 0
y_msm = np.zeros((num_stages, num_lines + 1))  # save the optimal config

for v in m.getVars():
    if v.X >= PRINT_THRESHOLD:
        print('{}: {:.3f}'.format(v.VarName, v.X))
    if "x_" in v.VarName:
        # x_s{}_{}_{}
        #   stage l' l
        str_temp = v.VarName.split('_')
        k = int(str_temp[1][1:])
        l = int(str_temp[2])
        lp = int(str_temp[3])
        op_cost_msm += 2 * alpha * cost[l, lp] * v.X * T_le_stage_prob[k]
        #                                           note: weighted by prob

    if "y_" in v.VarName:
        # y_s{}_{}
        #   stage l
        str_temp = v.VarName.split('_')
        k = int(str_temp[1][1:])
        l = int(str_temp[2])
        y_msm[k, l] = v.X

total_cost_msm = obj.getValue()
user_cost_msm = total_cost_msm - op_cost_msm
print('msm model estimated user cost: {:.3f}'.format(user_cost_msm))
print('msm model estimated operator cost: {:.3f}'.format(op_cost_msm))
print('msm model estimated total cost: {:.3f}'.format(total_cost_msm))


# prepare evaluation input: y[e, l]
# this matrix stores the actions of operators assuming that disruption is ongoing
# RMK: there is no need to accumulate operator cost
#      since operator cost is the same with model output.
y_eval = np.zeros((num_evals, num_lines + 1))
for k in range(num_stages):
    for e in range(stage_start_eval_time[k], stage_end_eval_time[k]):
        for l in range(num_lines + 1):
            y_eval[e, l] = y_msm[k, l]
# y_n is the configuration of fleets when disruption is over
y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends
model_name = 'msm'

# call evaluation function
user_cost_msm_eval = evaluation.evaluation(model_name, y_eval,
           y_n, num_evals, line_capacity,
           num_segments, segment_line, segment_cost,
           path_segment_incidence, boarding_flag,
           num_ods, od_num_paths, od_path2index, Q_evals,
           T_rd, T_prob_mass, eval_interval, M)
total_cost_msm_eval = user_cost_msm_eval + op_cost_msm
print("msm model evaluated total user cost:", user_cost_msm_eval)
print('msm model evaluated total cost: {:.3f}'.format(total_cost_msm_eval))


















# -------------- initialization time model (ITM)------------------
print("\n\n\n\n\n\n\ninitializtation time model (ITM) starts...")

# vars are:
# p_d, p_r, p_n path choice;
# y - fleet change
# x - relocation decisions
# z - initialization time; we assume that z only happens at eval time point for simplicity and computation efficiency
# this could easily be relaxed.
#
# where
# 'd' means disrupted, no relocation, like lla
# 'r' means disrupted, with relocation, like bm
# 'n' means normal, nm
#
# 'e' for disruption ending time; disruption end at the ending of the interval;
# hence disruption lasts for at least one interval (e = 0);
# 'z' for initialization time; service is initialized at the beginning of that interval
# Note: initialized at the start of z vs disruption ended at the end of e
#
# if e >= z, relocation will start
#   during time [0, z), user experience lla case
#   during [z, e], user experience bm case
#   during (e, T_eval_ub), user experience nm case
# if e < z, relocation not started
#   during time [0, e], user experience lla case
#   during time (e, T_eval_ub), user experience nm case
#
# For the purpose of optimization, e is unknown, hence E[e|z] will be used for the estimation of ending time

# compute disrupted, no adjustment path travel time (lla)
# compute segbc_d (segment boarding cost)  Note: constants
# y0_p is used for computation
segbc_d = np.zeros(num_segments)
for seg in range(num_segments):
    tmp_line = segment_line[seg]
    if y_0p[tmp_line] > 0:
        segbc_d[seg] = T_rd[tmp_line] / y_0p[tmp_line] / 2
        #                               y_0p initial fleet after disruption
    else:
        segbc_d[seg] = M

# compute pathc_d (path cost)  Note: constants
pathc_d = np.zeros(num_paths)
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
            #        use lla
            path_id = od_path2index[od][path]
            tmp_ttc = 0 # total cost accu
            #    this is not just traveling time
            for seg in range(num_segments):
                if boarding_flag[path_id, seg] == 1:  # Note: here don't use path_segment_incidence!
                    tmp_ttc += segbc_d[seg]
                if path_segment_incidence[path_id, seg] == 1:
                    tmp_ttc += segment_cost[seg]
            pathc_d[path_id] = tmp_ttc

# compute normal path travel time (nm)
# compute segbc_n (segmen boarding cost)  Note: constants
# compute pathc_n (path cost)
# p_0s is used for computation of these values
# these two (segbc_n, pathc_n) have been calculated in multistage model;
# these variables still available here; hence not repeated.

# optimization
# vars to update over loops
total_cost_itm = M
user_cost_itm = M
op_cost_itm = M
optimal_z_itm = M  # optimal initialization time
y_itm = [0] * (num_lines + 1)  # optimal fleet size setting

y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends

# iterate over all possible initialization time
# we only consider [0, 2/3 * num_evals] to speed up
# Note: initialized at the start of z vs disruption ended at the end of e
for z in range(MAX_INITIALIZATION_TIME):
    print("\n\ncurrent z:", z)
    m.reset()
    m = gp.Model("m")
    m.Params.LogToConsole = 0
    m.setParam('TimeLimit', GUROBI_TIME_LIMIT_ITM)

    # add variables
    # syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
    # add p_d, p_r, p_n
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            # if z > 0:
            if path_avail_lla[od][path] == 1:
                #        use lla
                exec("p_d_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_d_{}_{}')".format(od, path, od, path), globals())
            if path_avail_bm[od][path] == 1:
                #        use bm
                exec("p_r_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_r_{}_{}')".format(od, path, od, path), globals())
            if path_avail_nm[od][path] == 1:
                #        use nm
                exec("p_n_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_n_{}_{}')".format(od, path, od, path), globals())

    # we only need to add y and x for 'r'
    # add y_r
    for line in range(num_lines + 1):
        if line == 0:
            exec("y_r_{} = m.addVar(lb=0.0, ub={}, name='y_r_{}')".format(line, y_ub_disrupt[line], line), globals())
        else:
            exec("y_r_{} = m.addVar(lb={}, ub={}, name='y_r_{}')".format(line, Y_LOWER_BOUND, y_ub_disrupt[line], line), globals())

    # add x_r
    for l in range(num_lines + 1):
        for lp in range(num_lines + 1):
            if l == 0 or lp == 0:
                exec("x_r_{}_{} = m.addVar(lb=0.0, ub=0.0, name='x_r_{}_{}')".format(l, lp, l, lp), globals())
            else:
                exec("x_r_{}_{} = m.addVar(lb=0.0, name='x_r_{}_{}')".format(l, lp, l, lp), globals())

    # add segbc_r var (segment boarding cost) for temporary use
    # only LB is specified, although we can specify a large UB
    # Note: no need to add var for 'd', 'n' - they are constants
    for seg in range(num_segments):
        exec("segbc_r_{} = m.addVar(lb=0.0, name='segbc_r_{}')".format(seg, seg))

    # add pathc_r var (path cost) for temporary use
    # only LB is specified, although we can specify a large UB
    # Note: no need to add var for 'd', 'n' - they are constants
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #       use 'bm'
                path_id = od_path2index[od][path]
                exec("pathc_r_{} = m.addVar(lb=0.0, name='pathc_r_{}')".format(path_id, path_id))

    # add constraints
    # add OD flow conservation constraints for 'd', 'r', and 'n'
    # syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
    for od in range(num_ods):
        # 'd'
        #if z > 0:
        const_str = ""
        count = 0
        for path in range(od_num_paths[od]):
            if path_avail_lla[od][path] == 1:
                #        use lla
                if count != 0:
                    const_str += " + "
                const_str += "p_d_{}_{}".format(od, path)
                count += 1
        const_str += " == 1.0"
        const_str = "m.addConstr(" + const_str + ", 'od_d_{}')".format(od)
        exec(const_str, globals())

        # 'r'
        const_str = ""
        count = 0
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #        use bm
                if count != 0:
                    const_str += " + "
                const_str += "p_r_{}_{}".format(od, path)
                count += 1
        const_str += " == 1.0"
        const_str = "m.addConstr(" + const_str + ", 'od_r_{}')".format(od)
        exec(const_str, globals())

        # 'n'
        const_str = ""
        count = 0
        for path in range(od_num_paths[od]):
            if path_avail_nm[od][path] == 1:
                # use nm
                if count != 0:
                    const_str += " + "
                const_str += "p_n_{}_{}".format(od, path)
                count += 1
        const_str += " == 1.0"
        const_str = "m.addConstr(" + const_str + ", 'od_n_{}')".format(od)
        exec(const_str, globals())

    # add capacity constraints for 'd', 'r', 'n'
    # syntax: m.addConstr( x + 2 * y + 3 * z >= 4, "c0")
    for seg in range(num_segments):
        # 'd'
        #if z > 0:
        # if you decide to start relocation immediately, then
        # 'd' case doesn't exist;
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                if path_avail_lla[od][path] == 1:
                    #        use lla
                    path_id = od_path2index[od][path]
                    if path_segment_incidence[path_id][seg] == 1:
                        if count != 0:
                            const_str += " + "
                        const_str += "{:.3f}".format(Q_evals_accu_exclude[od, z])
                        #                  use Q_evals_accu_exclude, demand in [0, z)
                        const_str += " * "
                        const_str += "p_d_{}_{}".format(od, path)
                        count += 1
        if const_str == "":
            continue
        const_str += " - "
        tmp_line = segment_line[seg]
        tmp_coeff = z * eval_interval * y_0p[tmp_line] / T_rd[tmp_line] * line_capacity[tmp_line]
        #          capacity in [0, z)
        const_str += "{:.3f}".format(tmp_coeff)
        const_str += " <= 0"
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_d_{}')".format(seg)
        exec(const_str, globals())

        # 'r'
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                if path_avail_bm[od][path] == 1:
                    #      use bm
                    path_id = od_path2index[od][path]
                    if path_segment_incidence[path_id][seg] == 1:
                        if count != 0:
                            const_str += " + "
                        const_str += "{:.3f}".format(Q_evals_between_include[od, z, T_expected_conditioning_eval[z]])
                        #                       demand in [z, E[e|z]]
                        # eg. z = 23, E[e|z] = 23, means at eval time 23, it's expected that disruption will end at end of eval time 23 
                        # then the disruption last for 1 eval interval
                        const_str += " * "
                        const_str += "p_r_{}_{}".format(od, path)
                        count += 1
        if const_str == "":
            continue
        const_str += " - "
        tmp_line = segment_line[seg]
        tmp_coeff = (T_expected_conditioning_eval[z] + 1 - z) * eval_interval / T_rd[tmp_line] * line_capacity[tmp_line]
        # interval [z, E[e|z]] len = E[e|z] + 1 - z
        const_str += "{:.3f} * y_r_{}".format(tmp_coeff, tmp_line)
        #                              capacity in  [z, E[e|z]]
        const_str += " <= 0"
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_r_{}')".format(seg)
        exec(const_str, globals())

        # 'n'
        const_str = ""
        count = 0
        for od in range(num_ods):
            for path in range(od_num_paths[od]):
                if path_avail_nm[od][path] == 1:
                    #        use nm
                    path_id = od_path2index[od][path]
                    if path_segment_incidence[path_id][seg] == 1:
                        if count != 0:
                            const_str += " + "
                        const_str += "{:.3f}".format(Q_evals_left_exclude[od, T_expected_conditioning_eval[z]])
                        #                           demand in (e, num_evals)  Note: don't use T_ub=240; use T_eval_ub=24
                        const_str += " * "
                        const_str += "p_n_{}_{}".format(od, path)
                        count += 1
        if const_str == "":
            continue
        const_str += " - "
        tmp_line = segment_line[seg]
        tmp_coeff = (T_eval_ub - 1 - T_expected_conditioning_eval[z]) * eval_interval * y_n[tmp_line] / T_rd[tmp_line] * line_capacity[tmp_line]
        #    capacity in (e, T_eval_ub)  len = T_eval_ub - 1 - e                    y_n
        const_str += "{:.3f}".format(tmp_coeff)
        const_str += " <= 0"
        const_str = "m.addConstr(" + const_str + ", 'seg_cap_n_{}')".format(seg)
        exec(const_str, globals())

    # add constraints on y_r
    # metro fleet
    const_str = "y_r_1 + y_r_4 + y_r_5 + y_r_6 == {}".format(num_metro_vehs)
    const_str = "m.addConstr(" + const_str + ", 'y_r_metro')"
    exec(const_str, globals())
    # bus fleet
    const_str = "y_r_2 + y_r_3 + y_r_7 + y_r_8 == {}".format(num_bus_vehs)
    const_str = "m.addConstr(" + const_str + ", 'y_r_bus')"
    exec(const_str, globals())
    # L7 and L2 will interfere; hence we add a constraint on y[1] + y[6] <= L2 capacity
    const_str = "y_r_1 + y_r_6 <= {}".format(y_ub_disrupt[1])
    const_str = "m.addConstr(" + const_str + ", 'y_r_metro_overlap')"
    exec(const_str, globals())

    # add constraints on x_r
    for l in range(num_lines + 1):

        const_str = ""
        count = 0
        for lp in range(num_lines + 1):
            if count != 0:
                const_str += " + "
            const_str += "x_r_{}_{}".format(l, lp)
            count += 1
        for lp in range(num_lines + 1):
            const_str += " - x_r_{}_{}".format(lp, l)
        const_str += " + y_r_{}".format(l)
        const_str += " == {}".format(y_0p[l])
        #                       use y_0p
        const_str = "m.addConstr(" + const_str + ", 'x_conservation_r_{}')".format(l)
        exec(const_str, globals())

    for l in range(num_lines + 1):
        const_str = "x_r_{}_{} == 0".format(l, l)
        const_str = "m.addConstr(" + const_str + ", 'x_r_ll=0_{}')".format(l)
        exec(const_str, globals())

    # compute segbc_r (segment boarding cost)
    for seg in range(num_segments):
        tmp_line = segment_line[seg]
        if y_ub_disrupt[tmp_line] > 0:  # if you don't add this condition, the problem would be infeasible!
            const_str = "segbc_r_{} * y_r_{}".format(seg, tmp_line)
            const_str += " >= {:.3f}".format(T_rd[tmp_line] / 2)
            # Note: this is nonlinear constraint! (obj still non-linear)
        else:
            const_str = "segbc_r_{}".format(seg)
            const_str += " == {:.3f}".format(M)
        const_str = "m.addConstr(" + const_str + ", 'segbc_r_{}')".format(seg)
        exec(const_str, globals())

    # compute pathc_r (path cost)
    for od in range(num_ods):
        for path in range(od_num_paths[od]):
            if path_avail_bm[od][path] == 1:
                #       use bm
                path_id = od_path2index[od][path]
                const_str = "pathc_r_{}".format(path_id)
                tmp_tc = 0 # travel cost accu
                for seg in range(num_segments):
                    if boarding_flag[path_id, seg] == 1:  # Note: here don't use path_segment_incidence!
                        const_str += " - segbc_r_{}".format(seg)
                    if path_segment_incidence[path_id, seg] == 1:
                        tmp_tc += segment_cost[seg]
                const_str += " == {:.3f}".format(tmp_tc)
                const_str = "m.addConstr(" + const_str + ", 'pathc_r_{}')".format(path_id)
                exec(const_str, globals())

    # add objective
    # syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
    #         m.setObjective(obj)
    obj_str = "obj = "
    count = 0
    # user cost terms
    # the disruption ends at end of eval interval e; the cost will be weighted by Prob(e)
    for e in range(num_evals):
        # iterate over all cases of e
        if e < z:
            # if the disruption ends before initialization, add up
            # 1) the user cost (disrupted) in [0, e]
            # 2) the user cost (undisrupted) in (e, T_eval_ub)
            for od in range(num_ods):
                for path in range(od_num_paths[od]):
                    path_id = od_path2index[od][path]
                    # terms are like: OD_flow * path cost * path_ratio
                    # Q[od] * p_od_path * path_cost[od_path2index[od][path]]

                    # part 1)
                    if count != 0:
                        obj_str += " + "
                    if path_avail_lla[od][path] == 1:
                        #        use lla
                        tmp_coeff = Q_evals_accu_include[od, e] * pathc_d[path_id] * T_prob_mass[e]
                        #           demand in [0, e]                pathc_d
                        obj_str += "{:.3f} * p_d_{}_{}".format(tmp_coeff, od, path)
                        count += 1

                    # part 2)
                    if count != 0:
                        obj_str += " + "
                    if path_avail_nm[od][path] == 1:
                        #        use nm
                        tmp_coeff = Q_evals_left_exclude[od, e] * pathc_n[path_id] * T_prob_mass[e]
                        #           demand in (e, T_eval_ub       pathc_n
                        obj_str += "{:.3f} * p_n_{}_{}".format(tmp_coeff, od, path)
                        count += 1
        else:  # e >= z, the initiated scenario last for at least one eval interval
            # if the disruption ends after initialization, add up
            # 1) the user cost (disrupted) in [0, z)
            # 2) the user cost (optimized) in [z, e]
            # 3) the user cost (undisrupted) in (e, T_eval_ub)
            for od in range(num_ods):
                for path in range(od_num_paths[od]):
                    path_id = od_path2index[od][path]
                    # terms are like: OD_flow * path cost * path_ratio
                    # Q[od] * p_od_path * path_cost[od_path2index[od][path]]

                    # part 1)
                    if count != 0:
                        obj_str += " + "
                    if path_avail_lla[od][path] == 1:
                        #        use lla
                        tmp_coeff = Q_evals_accu_exclude[od, z] * pathc_d[path_id] * T_prob_mass[e]
                        #           demand in [0, z)              pathc_d
                        obj_str += "{:.3f} * p_d_{}_{}".format(tmp_coeff, od, path)
                        count += 1

                    # part 2)
                    if count != 0:
                        obj_str += " + "
                    if path_avail_bm[od][path] == 1:
                        #        use bm
                        tmp_coeff = Q_evals_between_include[od, z, e] * T_prob_mass[e]
                        #       demand in [z, e]
                        obj_str += "{:.3f} * p_r_{}_{} * pathc_r_{}".format(tmp_coeff, od, path, path_id)
                        count += 1

                    # part 3)
                    if count != 0:
                        obj_str += " + "
                    if path_avail_nm[od][path] == 1:
                        #      use nm
                        tmp_coeff = Q_evals_left_exclude[od, e] * pathc_n[path_id] * T_prob_mass[e]
                        #           demand in (e, T_ub)         pathc_n
                        obj_str += "{:.3f} * p_n_{}_{}".format(tmp_coeff, od, path)
                        count += 1

    # operator cost:
    for l in range(num_lines + 1):
        for lp in range(num_lines + 1):
            tmp_coeff = 2 * alpha * cost[l, lp] * T_le_eval_prob[z]
            obj_str += " + {:.3f} * x_r_{}_{}".format(tmp_coeff, l, lp)
    exec(obj_str, globals())
    exec("m.setObjective(obj)", globals())

    m.params.NonConvex = 2
    m.optimize()

    # update itr vars if needed
    if obj.getValue() < total_cost_itm:
        total_cost_itm = obj.getValue()
        optimal_z_itm = z
        print("optimal_z updated to:", z)
        print("total_cost_itm updated to:", total_cost_itm)

        # operator cost
        op_cost_itm = 0
        for v in m.getVars():
            if v.X >= PRINT_THRESHOLD:
                print('{}: {:.3f}'.format(v.VarName, v.X))
            if "x_" in v.VarName:
                # x_r_{}_{}
                #   stage l' l
                str_temp = v.VarName.split('_')
                l = int(str_temp[2])
                lp = int(str_temp[3])
                op_cost_itm += 2 * alpha * cost[l, lp] * v.X * T_le_eval_prob[z]
        # use cost
        user_cost_itm = total_cost_itm - op_cost_itm

        # y settings
        for v in m.getVars():
            if "y_" in v.VarName:
                # y_r_{}
                str_temp = v.VarName.split('_')
                l = int(str_temp[2])
                y_itm[l] = v.X

# print result at the end
print("\n\nthe optimal initialization time (z) is:", optimal_z_itm)
print('itm model estimated user cost: {:.3f}'.format(user_cost_itm))
print('itm model estimated operator cost: {:.3f}'.format(op_cost_itm))
print('itm model estimated total cost: {:.3f}'.format(total_cost_itm))


# prepare evaluation input: y[e, l]
# this matrix stores the actions of operators assuming that disruption is ongoing
# RMK: there is no need to accumulate operator cost
#      since operator cost is the same with model output.
y_eval = np.zeros((num_evals, num_lines + 1))
for e in range(optimal_z_itm):
    for l in range(num_lines + 1):
        y_eval[e, l] = y_0p[l]
for e in range(optimal_z_itm, num_evals):
    for l in range(num_lines + 1):
        y_eval[e, l] = y_itm[l]
# y_n is the configuration of fleets when disruption is over
y_n = y_0s # we assume that the fleet size go back to y_0 when disruption ends
model_name = 'itm'

# call evaluation function
user_cost_itm_eval = evaluation.evaluation(model_name, y_eval,
                      y_n, num_evals, line_capacity,
                      num_segments, segment_line, segment_cost,
                      path_segment_incidence, boarding_flag,
                      num_ods, od_num_paths, od_path2index, Q_evals,
                      T_rd, T_prob_mass, eval_interval, M)
total_cost_itm_eval = user_cost_itm_eval + op_cost_itm
print("itm model evaluated total user cost:", user_cost_itm_eval)
print('itm model evaluated total cost: {:.3f}'.format(total_cost_itm_eval))

print("\nfinished!")

print("\n\nargs:")
print("Q_scenario:", Q_scenario)
print("q_0:", q_0)
print("q_max:", q_max)
print("T_scenario:", T_scenario)
print("num_stages:", num_stages)
print("alpha:", alpha)

print("\nsummary of of all methods results:")
print("     |          mode estimation                 |      model evaluation")
print("model| user_cost_est  op_cost  total_cost_est | user_cost_eval  total_cost_eval")
print("lla", user_cost_lla*VOT, op_cost_lla*VOT, total_cost_lla*VOT, user_cost_lla_eval*VOT, total_cost_lla_eval*VOT)
print("bb", user_cost_bb*VOT, op_cost_bb*VOT, total_cost_bb*VOT, user_cost_bb_eval*VOT, total_cost_bb_eval*VOT)
print("bm", user_cost_bm*VOT, op_cost_bm*VOT, total_cost_bm*VOT, user_cost_bm_eval*VOT, total_cost_bm_eval*VOT)
print("msm", user_cost_msm*VOT, op_cost_msm*VOT, total_cost_msm*VOT, user_cost_msm_eval*VOT, total_cost_msm_eval*VOT)
print("itm", user_cost_itm*VOT, op_cost_itm*VOT, total_cost_itm*VOT, user_cost_itm_eval*VOT, total_cost_itm_eval*VOT)
print("\nRemark: 1) the optimal number of backup buses used in bb:", optimal_num_bb_bus)
print("        2) the optimal initialization time:", optimal_z_itm)