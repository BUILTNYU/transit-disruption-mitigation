import numpy as np


def cost_generation(M, BLT, BBT, MLT, MST):
    # M # big-M
    # BLT # bus line transfer cost
    # BBT # bus back-up transfer cost
    # MLT # metro line transfer cost
    # MST # metro short-turn cost
    #                   0      1    2     3      4    5     6      7     8
    #                   L1    L2    L3    L4    L5    L6    L7    L8    BKUP
    cost = np.array([[   0,  MLT,    M,    M,  MST,  MST,  MLT,    M,    M], # L1 0
                     [   0,    0,    M,    M,  MLT,  MLT,  MLT,    M,    M], # L2 1
                     [   0,    0,    0,  BLT,    M,    M,    M,  BLT,  BBT], # L3 2
                     [   0,    0,    0,    0,    M,    M,    M,  BLT,  BBT], # L4 3
                     [   0,    0,    0,    0,    0,  MLT,  MLT,    M,    M], # L5 4
                     [   0,    0,    0,    0,    0,    0,  MLT,    M,    M], # L6 5
                     [   0,    0,    0,    0,    0,    0,    0,    M,    M], # L7 6
                     [   0,    0,    0,    0,    0,    0,    0,    0,  BBT], # L8 7
                     [   0,    0,    0,    0,    0,    0,    0,    0,    0]]) # BKUP 8
    # Note:
    # 1) metro has no back up vehicles
    # 2) transfer between metro and bus is forbidden
    # 3) we first fill the upper half

    # complete the other half of the matrix
    for i in range(0, 9):
        for j in range(0, i):
            cost[i, j] = cost[j, i]

    return cost