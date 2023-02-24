# Deterministic disruption duration case

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from generate_cost import *
from generate_path import *
from generate_segment import *

print("deterministic model starts...")
print("setting up...")

# -----------------------  parameters -------------------------------
alpha = 10.0 # weight of user cost and operator cost; 1 / value of time per minute
T = 60 # disruption duration
T_ub = 240 # upper bound for disruption duration
num_lines = 8
M = 100000000 # big-M
GUROBI_TIME_LIMIT = 5 * 60 # time limit for gurobi (s)

# operator cost settings
BLT = 100 # bus line transfer cost
BBT = 300 # bus back-up transfer cost
MLT = 200 # metro line transfer cost
MST = 0 # metro short-turn cost
# operator cost matrix
# (num_lines + 1) x (num_lines + 1) matrix, since backup is included
# backup is the 9-th "line" (index 8)
c = cost_generation(M, BLT, BBT, MLT, MST)

# round trip time of lines
#       0   1   2   3   4   5  6   7
#       L1  L2  L3  L4  L5  L6 L7  L8
T_rd = [36, 36, 96, 96, 20, 8, 36, 16]
beta = 1.0  # capacity constraints coefficients
capacity_metro = 1000  # capacity of each train
capacity_bus = 100  # capacity of each bus

# demand during [0, T]
num_ods = 8
q_0 = 10 * np.ones(num_ods)   # 12 is good
q_max = 1.25 * q_0
# total demand in [0, T] for each OD
Q = np.zeros(num_ods)
for i in range(num_ods):
    Q[i] = -4/3/(T_ub**2)*(q_max[i] - q_0[i])*(T**3) + 2/T_ub*(q_max[i] - q_0[i])*(T**2) + q_0[i]*T

# initial fleet
#      0  1  2  3   4  5  6  7  8
#      L1 L2 L3 L4  L5 L6 L7 L8 backup_bus
y_0 = [3, 3, 12, 12, 0, 0, 0, 0, 2]
num_metro_vehs = y_0[0] + y_0[1]
num_bus_vehs = y_0[2] + y_0[3] + y_0[8]

# immediately after disruption, y_0'
#       L1 L2 L3  L4  L5 L6 L7 L8 backup_bus
y_0p = [0, 3, 12, 12, 2, 1, 0, 0, 2]
# L1 is no longer available

# line capacity generation
#                      L1              L2             L3           L4            L5               L6              L7             L8
line_capacity = [capacity_metro, capacity_metro, capacity_bus, capacity_bus, capacity_metro, capacity_metro, capacity_metro, capacity_bus]

# segment generation
num_segments, segment_cost, segment_line = segment_generation()

# path enumeration
# if a passenger has not been disrupted; his will only be given the same path as normal case
# path segment incidence matrix
num_paths, path_segment_incidence, boarding_flag, \
    od_num_paths, od_path2index, path_avail_lla, path_avail_bb = path_enumeration(num_segments)


# -------------- local level adjust model (LLA)--------------
# just compute the user cost
# operator cost is zero since back-up bus is not used and no fleet rellocation
print("\nlla model starts...")

# the fleet size is just y_0p
y = y_0p

# ----------- repeated code start----------------\
# segment capacity
segment_capacity = np.zeros(num_segments)
for i in range(num_segments):
    temp_line = segment_line[i]
    segment_capacity[i] = T / T_rd[temp_line] * y[temp_line] * line_capacity[temp_line]

# segments boarding cost
segment_boarding_cost = np.zeros(num_segments)
for seg in range(num_segments):
    temp_line = segment_line[seg]
    if y[temp_line] > 0:
        segment_boarding_cost[seg] = T_rd[temp_line] / y[temp_line] / 2
    else:
        segment_boarding_cost[seg] = M

# path cost
path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost))\
            + np.dot(boarding_flag, np.transpose(segment_boarding_cost))
# optimize path flow choice under capacity constraints
m = gp.Model("m")

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
var_index_in_p = []
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())
            var_index_in_p.append(count)
        count += 1

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_lla[od][path] == 1:
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
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q[od])
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
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
            path_id = od_path2index[od][path]
            temp_coeff = Q[od] * path_cost[path_id]
            obj_str += "{:.3f} * p_{}_{}".format(temp_coeff, od, path)
            count += 1
exec(obj_str)
exec("m.setObjective(obj)", globals())

m.optimize()
# ----------- repeated code end----------------/

# store results
p_lla = np.zeros(num_paths) # the user path choice p
count = 0
for v in m.getVars():
    print('{}: {:.3f}'.format(v.VarName, v.X))
    p_lla[var_index_in_p[count]] = v.X
    count += 1

# use cost
user_cost_lla = obj.getValue()
print('user cost (obj): {:.3f}'.format(user_cost_lla))


# -------------- bus bridging model (BB) ---------------------

# ------ case 1: use 1 bus to bridge
print("\nbb model (case 1) starts...")

# update fleet
#         L1...        L5 L6 L7 L8 backup_bus
y = [0, 3, 12, 12, 2, 1, 0, 1, 1]

# ----------- repeated code start----------------\
# segment capacity
segment_capacity = np.zeros(num_segments)
for i in range(num_segments):
    temp_line = segment_line[i]
    segment_capacity[i] = T / T_rd[temp_line] * y[temp_line] * line_capacity[temp_line]

# segments boarding cost
segment_boarding_cost = np.zeros(num_segments)
for i in range(num_segments):
    if y[segment_line[i]] > 0:
        segment_boarding_cost[i] = T_rd[segment_line[i]] / y[segment_line[i]] / 2
    else:
        segment_boarding_cost[i] = M

# path cost
path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
            + np.dot(boarding_flag, np.transpose(segment_boarding_cost))

# optimize path flow choice under capacity constraints
m = gp.Model("m")

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
var_index_in_p = []
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bb[od][path] == 1:  # change!!
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())
            var_index_in_p.append(count)
        count += 1

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_bb[od][path] == 1:  # change!!
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
            if path_avail_bb[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:  # change!!
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q[od])
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
        if path_avail_bb[od][path] == 1:  # change!!
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
            path_id = od_path2index[od][path]
            temp_coeff = Q[od] * path_cost[path_id]
            obj_str += "{:.3f} * p_{}_{}".format(temp_coeff, od, path)
            count += 1
exec(obj_str)
exec("m.setObjective(obj)", globals())

m.optimize()
# ----------- repeated code end----------------/

# store results
p_bb_1 = np.zeros(num_paths) # the user path choice p
count = 0
for v in m.getVars():
    print('{}: {:.3f}'.format(v.VarName, v.X))
    p_bb_1[var_index_in_p[count]] = v.X
    count += 1

# use cost
user_cost_bb_1 = obj.getValue()
print('user cost (obj): {:.3f}'.format(user_cost_bb_1))
op_cost_bb_1 = 2 * y[7] * c[8, 7] # relocate from backup to L8
print('operator cost: {:.3f}'.format(op_cost_bb_1))
total_cost_bb_1 = user_cost_bb_1 + alpha * op_cost_bb_1
print('total: {:.3f}'.format(total_cost_bb_1))


# ------ case 2: use 2 bus to bridge
print("\nbb model (case 2) starts...")

# update fleet
#    L1...        L5 L6 L7 L8 backup_bus
y = [0, 3, 12, 12, 2, 1, 0, 2, 0]

# ----------- repeated code start----------------\
# segment capacity
segment_capacity = np.zeros(num_segments)
for i in range(num_segments):
    temp_line = segment_line[i]
    segment_capacity[i] = T / T_rd[temp_line] * y[temp_line] * line_capacity[temp_line]

# segments boarding cost
segment_boarding_cost = np.zeros(num_segments)
for i in range(num_segments):
    if y[segment_line[i]] > 0:
        segment_boarding_cost[i] = T_rd[segment_line[i]] / y[segment_line[i]] / 2
    else:
        segment_boarding_cost[i] = M

# path cost
path_cost = np.dot(path_segment_incidence, np.transpose(segment_cost)) \
            + np.dot(boarding_flag, np.transpose(segment_boarding_cost))

# optimize path flow choice under capacity constraints
m = gp.Model("m")

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
var_index_in_p = []
count = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if path_avail_bb[od][path] == 1:  # change!!
            exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())
            var_index_in_p.append(count)
        count += 1

# add OD flow conservation constraints
# syntax: m.addConstr( x * x + 2 == - x + 4, "intersection equality")
for od in range(num_ods):
    const_str = ""
    count = 0
    for path in range(od_num_paths[od]):
        if path_avail_bb[od][path] == 1:  # change!!
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
            if path_avail_bb[od][path] == 1 and path_segment_incidence[path_id][seg] == 1:  # change!!
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q[od])
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
        if path_avail_bb[od][path] == 1:  # change!!
            if count != 0:
                obj_str += " + "
            # terms are like: OD_flow * path cost * path_ratio
            # Q[od] * path_cost[od_path2index[od][path]] * p_0_0
            path_id = od_path2index[od][path]
            temp_coeff = Q[od] * path_cost[path_id]
            obj_str += "{:.3f} * p_{}_{}".format(temp_coeff, od, path)
            count += 1
exec(obj_str)
exec("m.setObjective(obj)", globals())

m.optimize()
# ----------- repeated code end----------------/

# store results
p_bb_2 = np.zeros(num_paths) # the user path choice p
count = 0
for v in m.getVars():
    print('{}: {:.3f}'.format(v.VarName, v.X))
    p_bb_2[var_index_in_p[count]] = v.X
    count += 1

# use cost
user_cost_bb_2 = obj.getValue()
print('user cost (obj): {:.3f}'.format(user_cost_bb_2))
op_cost_bb_2 = 2 * y[7] * c[8, 7] # relocate from backup to L8
print('operator cost: {:.3f}'.format(op_cost_bb_2))
total_cost_bb_2 = user_cost_bb_2 + alpha * op_cost_bb_2
print('total: {:.3f}'.format(total_cost_bb_2))


# choose between two cases:
user_cost_bb = 0.0
op_cost_bb = 0.0
total_cost_bb = 0.0
num_bb_buses = 0
if total_cost_bb_1 < total_cost_bb_2:
    user_cost_bb = user_cost_bb_1
    op_cost_bb = op_cost_bb_1
    total_cost_bb = total_cost_bb_1
    num_bb_buses = 1
else:
    user_cost_bb = user_cost_bb_2
    op_cost_bb = op_cost_bb_2
    total_cost_bb = total_cost_bb_2
    num_bb_buses = 2
print("\nsummarizing two cases:")
print('number of backup buses used for bridging: {}'.format(num_bb_buses))
print('user cost (obj): {:.3f}'.format(user_cost_bb))
print('operator cost: {:.3f}'.format(op_cost_bb))
print('total: {:.3f}'.format(total_cost_bb))


# -------------- basic model (BM)----------------------
print("\nbm model starts...")
# vars are:
# p - path choice, as before
# y - fleet change
# x - relocation decisions

# optimize path flow choice under capacity constraints
m = gp.Model("m")
m.setParam('TimeLimit', GUROBI_TIME_LIMIT)

# add variables
# syntax: x = m.addVar(lb=0.0, ub=1.0, name="x")
# add p
# Note that every path is possible under basic model
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        exec("p_{}_{} = m.addVar(lb=0.0, ub=1.0, name='p_{}_{}')".format(od, path, od, path), globals())

# add y
# the ub of y is
#       L1 L2 L3 L4  L5 L6 L7 L8 backup_bus
y_ub = [0, 6, 20, 20, 4, 2, 5, 3, 2]
for line in range(num_lines + 1):
    exec("y_{} = m.addVar(lb=0.0, ub={}, name='y_{}')".format(line, y_ub[line], line), globals())

# add x
for l in range(num_lines + 1):
    for lp in range(num_lines + 1):
        exec("x_{}_{} = m.addVar(lb=0.0, ub=1.0, name='x_{}_{}')".format(l, lp, l, lp), globals())

# add segbc var (segment boarding cost) for temporary use
# only LB is specified, although we can specify a large UB
for seg in range(num_segments):
    exec("segbc_{} = m.addVar(lb=0.0, name='segbc_{}')".format(seg, seg))

# add pathc var (path cost) for temporary use
# only LB is specified, although we can specify a large UB
for path in range(num_paths):
    exec("pathc_{} = m.addVar(lb=0.0, name='pathc_{}')".format(path, path))

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
            if path_segment_incidence[path_id][seg] == 1:
                if count != 0:
                    const_str += " + "
                const_str += "{:.3f}".format(Q[od])
                const_str += " * "
                const_str += "p_{}_{}".format(od, path)
                count += 1
    if const_str == "":
        continue
    const_str += " - "
    temp_line = segment_line[seg]
    const_str += "{:.3f} * y_{}".format(T / T_rd[temp_line] * line_capacity[temp_line], temp_line)
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
const_str = "y_1 + y_6 <= {}".format(y_ub[1])
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
    temp_line = segment_line[seg]
    const_str = "segbc_{} * y_{}".format(seg, temp_line)
    const_str += " >= {:.3f}".format(T_rd[temp_line] / 2)
    # Note: this is nonlinear constraint! (obj still non-linear)
    const_str = "m.addConstr(" + const_str + ", 'segbc_')".format(seg)
    exec(const_str, globals())

# compute pathc (path cost)
for path in range(num_paths):
    const_str = "pathc_{}".format(path)
    tc_temp = 0
    for seg in range(num_segments):
        if boarding_flag[path, seg] == 1:  # Note: here don't use path_segment_incidence!
            const_str += " - segbc_{}".format(seg)
        if path_segment_incidence[path, seg] == 1:
            tc_temp += segment_cost[seg]
    const_str += " == {:.3f}".format(tc_temp)
    const_str = "m.addConstr(" + const_str + ", 'pathc_{}')".format(path)
    exec(const_str, globals())

# add objective
# syntax: obj = x**2 + x*y + y**2 + y*z + z**2 + 2*x
#         m.setObjective(obj)
obj_str = "obj = "
count = 0
# user cost terms
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        if count != 0:
            obj_str += " + "
        # terms are like: OD_flow * path cost * path_ratio
        # Q[od] * p_od_path * path_cost[od_path2index[od][path]]
        path_id = od_path2index[od][path]
        obj_str += "{:.3f} * p_{}_{} * pathc_{}".format(Q[od], od, path, path_id)
        count += 1

# operator cost:
for l in range(num_lines + 1):
    for lp in range(num_lines + 1):
        obj_str += " + {:3f} * x_{}_{}".format(2 * alpha * c[l, lp], l, lp)

exec(obj_str)
exec("m.setObjective(obj)", globals())

m.params.NonConvex = 2
m.NumStart = 1
# set StartNumber
m.params.StartNumber = 1
# now set MIP start values using the Start attribute, e.g.:
count_y = 0
for v in m.getVars():
    if "y_" in v.VarName:
        v.Start = y_0p[count_y]
        count_y += 1
    if "x_" in v.VarName:
        v.Start = 0

m.optimize()

# store results
p_bm = []
pathc_bm = []
x_bm = []
for v in m.getVars():
    print('{}: {:.3f}'.format(v.VarName, v.X))
    if "p_" in v.VarName:
        p_bm.append(v.X)
    if "path" in v.VarName:
        pathc_bm.append(v.X)
    if "x_" in v.VarName:
        x_bm.append(v.X)
# use cost
user_cost_bm = 0
for od in range(num_ods):
    for path in range(od_num_paths[od]):
        path_id = od_path2index[od][path]
        user_cost_bm += Q[od] * p_bm[path_id] * pathc_bm[path_id]
print('user cost: {:.3f}'.format(user_cost_bm))
# operator cost
op_cost_bm = 0
count = 0
for l in range(num_lines + 1):
    for lp in range(num_lines + 1):
        op_cost_bm += 2 * c[l, lp] * x_bm[count]  # no alpha!
        if x_bm[count] > 0 and c[l, lp] > 10000:
            print("error: ", l, lp, count)
            print(c[l, lp], x_bm[count])
        count += 1
print('operator cost: {:.3f}'.format(op_cost_bm))
total_cost_bm = obj.getValue()
print('total: {:.3f}'.format(total_cost_bm))


