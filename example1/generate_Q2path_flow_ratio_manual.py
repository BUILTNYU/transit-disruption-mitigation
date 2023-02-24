def Q2path_flow_ratio_manual_generation():
    '''generate Q2path flow ratio manually'''
    # path choice
    # od flow to path flow transition matrix
    Q2path_flow_ratio = np.zeros((num_ods, num_paths))
    Q2path_flow_ratio[0, 0] = 1 # lla, bb, bm
    Q2path_flow_ratio[0, 1] = 0 # bb, bm
    Q2path_flow_ratio[0, 2] = 0 # bm

    Q2path_flow_ratio[1, 3] = 1 # lla, bb, bm
    Q2path_flow_ratio[1, 4] = 0 # lla, bb, bm
    Q2path_flow_ratio[1, 5] = 0 # lla, bb, bm
    Q2path_flow_ratio[1, 6] = 0 # bm

    Q2path_flow_ratio[2, 7] = 1 # lla, bb, bm
    Q2path_flow_ratio[2, 8] = 0 # lla, bb, bm

    Q2path_flow_ratio[3, 9] = 1 # bb, bm
    Q2path_flow_ratio[3, 10] = 0 # bm
    Q2path_flow_ratio[3, 11] = 0 # lla, bb, bm
    Q2path_flow_ratio[3, 12] = 0 # bm

    Q2path_flow_ratio[4, 13] = 1 # bm
    Q2path_flow_ratio[4, 14] = 0 # lla, bb, bm
    Q2path_flow_ratio[4, 15] = 0 # lla, bb, bm

    Q2path_flow_ratio[5, 16] = 1 # lla, bb, bm
    Q2path_flow_ratio[5, 17] = 0 # lla, bb, bm

    Q2path_flow_ratio[6, 18] = 1 # lla, bb, bm
    Q2path_flow_ratio[6, 19] = 0 # bm
    Q2path_flow_ratio[6, 20] = 0 # bb, bm
    Q2path_flow_ratio[6, 21] = 0 # lla, bb, bm
    Q2path_flow_ratio[6, 22] = 0 # lla, bb, bm

    Q2path_flow_ratio[7, 23] = 1 # lla, bb, bm
    Q2path_flow_ratio[7, 24] = 0 # bb, bm
    Q2path_flow_ratio[7, 25] = 0 # bm

    return Q2path_flow_ratio
