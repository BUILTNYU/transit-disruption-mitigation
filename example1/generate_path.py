import numpy as np

# Abbr:
# lla - local level model
# bb - bus bridging model
# bm - basic model
# nm - normal case


def path_enumeration(num_segments):
    ''' path enumeration '''
    # path segment incidence matrix
    num_paths = 26
    path_segment_incidence = np.zeros((num_paths, num_segments))
    boarding_flag = np.zeros((num_paths, num_segments))

    # i is the path_id
    i = 0
    # comment format:
    # path_id od path: Line>stop_seq (availability)
    # 0 OD>1-10 1: L2>1-5-6-10 (lla, bb, bm, nm)
    boarding_flag[i, 0] = 1
    path_segment_incidence[i, 0] = 1
    path_segment_incidence[i, 1] = 1
    path_segment_incidence[i, 2] = 1
    i += 1
    # 1 OD>1-10: 2: L5>1-5-9 L8>9-10 (bb, bm)
    boarding_flag[i, 32] = 1
    path_segment_incidence[i, 32] = 1
    path_segment_incidence[i, 33] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    i += 1
    #  2 OD>1-10 3: L7>1-5-6-10 (bm)
    boarding_flag[i, 38] = 1
    path_segment_incidence[i, 38] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    i += 1

    #  3 OD>5-14 1: L2>5-6-10-13 L3>13-14 (lla, bb, bm, nm)
    boarding_flag[i, 1] = 1
    path_segment_incidence[i, 1] = 1
    path_segment_incidence[i, 2] = 1
    path_segment_incidence[i, 3] = 1
    boarding_flag[i, 13] = 1
    path_segment_incidence[i, 13] = 1
    i += 1
    #  4 OD>5-14 2: L5>5-9 L3>9-12-13-14 (lla, bb, bm)
    boarding_flag[i, 33] = 1
    path_segment_incidence[i, 33] = 1
    boarding_flag[i, 11] = 1
    path_segment_incidence[i, 11] = 1
    path_segment_incidence[i, 12] = 1
    path_segment_incidence[i, 13] = 1
    i += 1
    #  5 OD>5-14 3: L2>5-6 L4>6-7-11-14 (lla, bb, bm, nm)
    boarding_flag[i, 1] = 1
    path_segment_incidence[i, 1] = 1
    boarding_flag[i, 23] = 1
    path_segment_incidence[i, 23] = 1
    path_segment_incidence[i, 24] = 1
    path_segment_incidence[i, 25] = 1
    i += 1
    #  6 OD>5-14 4: L7>5-6-10-11 L4>11-14 (bm)
    boarding_flag[i, 39] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    path_segment_incidence[i, 41] = 1
    boarding_flag[i, 25] = 1
    path_segment_incidence[i, 25] = 1
    i += 1

    #  7 OD>3-13 1: L4>3-6 L2>6-10-13 (lla, bb, bm, nm)
    boarding_flag[i, 22] = 1
    path_segment_incidence[i, 22] = 1
    boarding_flag[i, 2] = 1
    path_segment_incidence[i, 2] = 1
    path_segment_incidence[i, 3] = 1
    i += 1
    #  8 OD>3-13 2: L4>3-6-7-11-14 L3>14-13 (lla, bb, bm, nm)
    boarding_flag[i, 22] = 1
    path_segment_incidence[i, 22] = 1
    path_segment_incidence[i, 23] = 1
    path_segment_incidence[i, 24] = 1
    path_segment_incidence[i, 25] = 1
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    i += 1

    #  9 OD>8-11 1: L3>8-9 L8>9-10 L6>10-11 (bb, bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 11] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    boarding_flag[i, 36] = 1
    path_segment_incidence[i, 36] = 1
    i += 1
    # 10 OD>8-11 2: L3>8-9 L8>9-10 L7>10-11 (bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    boarding_flag[i, 41] = 1
    path_segment_incidence[i, 41] = 1
    i += 1
    # 11 OD>8-11 3: L3>8-9-12-13-14 L4>14-11 (lla, bb, bm, nm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    path_segment_incidence[i, 11] = 1
    path_segment_incidence[i, 12] = 1
    path_segment_incidence[i, 13] = 1
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    i += 1
    # 12 OD>8-11 4: L3>8-9 L5>9-5 L7>5-6-10-11 (bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    boarding_flag[i, 39] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    path_segment_incidence[i, 41] = 1
    i += 1

    # 13 OD>11-2 1: L7>11-10-6-5-1 L4>1-2 (bm)
    boarding_flag[i, 42] = 1
    path_segment_incidence[i, 42] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    path_segment_incidence[i, 45] = 1
    boarding_flag[i, 20] = 1
    path_segment_incidence[i, 20] = 1
    i += 1
    # 14 OD>11-2 2: L4>11-7-6-3-2 (lla, bb, bm, nm)
    boarding_flag[i, 27] = 1
    path_segment_incidence[i, 27] = 1
    path_segment_incidence[i, 28] = 1
    path_segment_incidence[i, 29] = 1
    path_segment_incidence[i, 30] = 1
    i += 1
    # 15 OD>11-2 3: L6>11-10 L2>10-6-5-1 L4>1-2 (lla,bb, bm)
    boarding_flag[i, 37] = 1
    path_segment_incidence[i, 37] = 1
    boarding_flag[i, 5] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    boarding_flag[i, 20] = 1
    path_segment_incidence[i, 20] = 1
    i += 1

    # 16 OD>13-4 1: L2>13-10-6-5-1 L3>1-4 (lla, bb, bm, nm)
    boarding_flag[i, 4] = 1
    path_segment_incidence[i, 4] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    boarding_flag[i, 8] = 1
    path_segment_incidence[i, 8] = 1
    i += 1
    # 17 OD>13-4 2: L3>13-12-9-8-4 (lla, bb, bm, nm)
    boarding_flag[i, 15] = 1
    path_segment_incidence[i, 15] = 1
    path_segment_incidence[i, 16] = 1
    path_segment_incidence[i, 17] = 1
    path_segment_incidence[i, 18] = 1
    i += 1

    # 18 OD>14-1 1: L3>14-13 L2>13-10-6-5-1 (lla, bb, bm, nm)
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    boarding_flag[i, 4] = 1
    path_segment_incidence[i, 4] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    i += 1
    # 19 OD>14-1 2: L4>14-11 L7>11-10-6-5-1 (bm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[19, 26] = 1
    boarding_flag[i, 42] = 1
    path_segment_incidence[i, 42] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    path_segment_incidence[i, 45] = 1
    i += 1
    # 20 OD>14-1 3: L4>14-11 L6>11-10 L8>10-9 L5>9-5-1 (bb, bm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    boarding_flag[i, 37] = 1
    path_segment_incidence[i, 37] = 1
    boarding_flag[i, 47] = 1
    path_segment_incidence[i, 47] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    path_segment_incidence[i, 35] = 1
    i += 1
    # 21 OD>14-1 4: L3>14-13-12-9-8-4-1 (lla, bb, bm, nm)
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    path_segment_incidence[i, 15] = 1
    path_segment_incidence[i, 16] = 1
    path_segment_incidence[i, 17] = 1
    path_segment_incidence[i, 18] = 1
    path_segment_incidence[i, 19] = 1
    i += 1
    # 22 OD>14-1 5: L4>14-11-7-6-3-2-1 (lla, bb, bm, nm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    path_segment_incidence[i, 27] = 1
    path_segment_incidence[i, 28] = 1
    path_segment_incidence[i, 29] = 1
    path_segment_incidence[i, 30] = 1
    path_segment_incidence[i, 31] = 1
    i += 1

    # 23 OD>10-5 1: L2>10-6-5 (lla, bb, bm, nm)
    boarding_flag[i, 5] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    i += 1
    # 24 OD>10-5 2: L8>10-9 L5>9-5 (bb, bm)
    boarding_flag[i, 47] = 1
    path_segment_incidence[i, 47] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    i += 1
    # 25 OD>10-5 3: L7>10-6-5 (bm)
    boarding_flag[i, 43] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    i += 1

    # number of paths that each OD has
    od_num_paths = [3, 4, 2, 4, 3, 2, 5, 3]
    od_path2index = [[0,1,2],
                     [3,4,5,6],
                     [7,8],
                     [9,10,11,12],
                     [13,14,15],
                     [16,17],
                     [18,19,20,21,22],
                     [23,24,25]]

    path_avail_lla = [[1,0,0],
                      [1,1,1,0],
                      [1,1],
                      [0,0,1,0],
                      [0,1,1],
                      [1,1],
                      [1,0,0,1,1],
                      [1,0,0]]

    path_avail_bb = [[1,1,0],
                     [1,1,1,0],
                     [1,1],
                     [1,0,1,0],
                     [0,1,1],
                     [1,1],
                     [1,0,1,1,1],
                     [1,1,0]]
    # Note: all paths are available to basic model
    return num_paths, path_segment_incidence, boarding_flag, od_num_paths, od_path2index, path_avail_lla, path_avail_bb


def path_enumeration_expanded(num_segments):
    ''' path enumeration with disrupted lines included'''
    # path segment incidence matrix
    num_paths = 32
    path_segment_incidence = np.zeros((num_paths, num_segments))
    boarding_flag = np.zeros((num_paths, num_segments))

    # i is the path_id
    i = 0
    # comment format:
    # path_id od path: Line>stop_seq (availability)
    # 0 OD>1-10 1: L2>1-5-6-10 (lla, bb, bm, nm)
    boarding_flag[i, 0] = 1
    path_segment_incidence[i, 0] = 1
    path_segment_incidence[i, 1] = 1
    path_segment_incidence[i, 2] = 1
    i += 1
    # 1 OD>1-10: 2: L5>1-5-9 L8>9-10 (bb, bm)
    boarding_flag[i, 32] = 1
    path_segment_incidence[i, 32] = 1
    path_segment_incidence[i, 33] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    i += 1
    #  2 OD>1-10 3: L7>1-5-6-10 (bm)
    boarding_flag[i, 38] = 1
    path_segment_incidence[i, 38] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    i += 1
    #  3 OD>1-10 4: L1>1-5-9-10 (nm)
    boarding_flag[i, 48] = 1
    path_segment_incidence[i, 48] = 1
    path_segment_incidence[i, 49] = 1
    path_segment_incidence[i, 50] = 1
    i += 1

    #  4 OD>5-14 1: L2>5-6-10-13 L3>13-14 (lla, bb, bm, nm)
    boarding_flag[i, 1] = 1
    path_segment_incidence[i, 1] = 1
    path_segment_incidence[i, 2] = 1
    path_segment_incidence[i, 3] = 1
    boarding_flag[i, 13] = 1
    path_segment_incidence[i, 13] = 1
    i += 1
    #  5 OD>5-14 2: L5>5-9 L3>9-12-13-14 (lla, bb, bm)
    boarding_flag[i, 33] = 1
    path_segment_incidence[i, 33] = 1
    boarding_flag[i, 11] = 1
    path_segment_incidence[i, 11] = 1
    path_segment_incidence[i, 12] = 1
    path_segment_incidence[i, 13] = 1
    i += 1
    #  6 OD>5-14 3: L2>5-6 L4>6-7-11-14 (lla, bb, bm, nm)
    boarding_flag[i, 1] = 1
    path_segment_incidence[i, 1] = 1
    boarding_flag[i, 23] = 1
    path_segment_incidence[i, 23] = 1
    path_segment_incidence[i, 24] = 1
    path_segment_incidence[i, 25] = 1
    i += 1
    #  7 OD>5-14 4: L7>5-6-10-11 L4>11-14 (bm)
    boarding_flag[i, 39] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    path_segment_incidence[i, 41] = 1
    boarding_flag[i, 25] = 1
    path_segment_incidence[i, 25] = 1
    i += 1
    #  8 OD>5-14 5: L1>5-9-11 L4>11-14 (nm)
    boarding_flag[i, 49] = 1
    path_segment_incidence[i, 49] = 1
    path_segment_incidence[i, 50] = 1
    path_segment_incidence[i, 51] = 1
    boarding_flag[i, 25] = 1
    path_segment_incidence[i, 25] = 1
    i += 1

    #  9 OD>3-13 1: L4>3-6 L2>6-10-13 (lla, bb, bm, nm)
    boarding_flag[i, 22] = 1
    path_segment_incidence[i, 22] = 1
    boarding_flag[i, 2] = 1
    path_segment_incidence[i, 2] = 1
    path_segment_incidence[i, 3] = 1
    i += 1
    # 10 OD>3-13 2: L4>3-6-7-11-14 L3>14-13 (lla, bb, bm, nm)
    boarding_flag[i, 22] = 1
    path_segment_incidence[i, 22] = 1
    path_segment_incidence[i, 23] = 1
    path_segment_incidence[i, 24] = 1
    path_segment_incidence[i, 25] = 1
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    i += 1

    # 11 OD>8-11 1: L3>8-9 L8>9-10 L6>10-11 (bb, bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 11] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    boarding_flag[i, 36] = 1
    path_segment_incidence[i, 36] = 1
    i += 1
    # 12 OD>8-11 2: L3>8-9 L8>9-10 L7>10-11 (bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    boarding_flag[i, 46] = 1
    path_segment_incidence[i, 46] = 1
    boarding_flag[i, 41] = 1
    path_segment_incidence[i, 41] = 1
    i += 1
    # 13 OD>8-11 3: L3>8-9-12-13-14 L4>14-11 (lla, bb, bm, nm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    path_segment_incidence[i, 11] = 1
    path_segment_incidence[i, 12] = 1
    path_segment_incidence[i, 13] = 1
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    i += 1
    # 14 OD>8-11 4: L3>8-9 L5>9-5 L7>5-6-10-11 (bm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    boarding_flag[i, 39] = 1
    path_segment_incidence[i, 39] = 1
    path_segment_incidence[i, 40] = 1
    path_segment_incidence[i, 41] = 1
    i += 1
    # 15 OD>8-11 5: L3>8-9 L1>9-10-11 (nm)
    boarding_flag[i, 10] = 1
    path_segment_incidence[i, 10] = 1
    boarding_flag[i, 50] = 1
    path_segment_incidence[i, 50] = 1
    path_segment_incidence[i, 51] = 1
    i += 1

    # 16 OD>11-2 1: L7>11-10-6-5-1 L4>1-2 (bm)
    boarding_flag[i, 42] = 1
    path_segment_incidence[i, 42] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    path_segment_incidence[i, 45] = 1
    boarding_flag[i, 20] = 1
    path_segment_incidence[i, 20] = 1
    i += 1
    # 17 OD>11-2 2: L4>11-7-6-3-2 (lla, bb, bm, nm)
    boarding_flag[i, 27] = 1
    path_segment_incidence[i, 27] = 1
    path_segment_incidence[i, 28] = 1
    path_segment_incidence[i, 29] = 1
    path_segment_incidence[i, 30] = 1
    i += 1
    # 18 OD>11-2 3: L6>11-10 L2>10-6-5-1 L4>1-2 (lla,bb, bm)
    boarding_flag[i, 37] = 1
    path_segment_incidence[i, 37] = 1
    boarding_flag[i, 5] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    boarding_flag[i, 20] = 1
    path_segment_incidence[i, 20] = 1
    i += 1
    # 19 OD>11-2 4: L1>11-10-9-5-1 L3>1-2 (nm)
    boarding_flag[i, 52] = 1
    path_segment_incidence[i, 52] = 1
    path_segment_incidence[i, 53] = 1
    path_segment_incidence[i, 54] = 1
    path_segment_incidence[i, 55] = 1
    boarding_flag[i, 20] = 1
    path_segment_incidence[i, 20] = 1
    i += 1

    # 20 OD>13-4 1: L2>13-10-6-5-1 L3>1-4 (lla, bb, bm, nm)
    boarding_flag[i, 4] = 1
    path_segment_incidence[i, 4] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    boarding_flag[i, 8] = 1
    path_segment_incidence[i, 8] = 1
    i += 1
    # 21 OD>13-4 2: L3>13-12-9-8-4 (lla, bb, bm, nm)
    boarding_flag[i, 15] = 1
    path_segment_incidence[i, 15] = 1
    path_segment_incidence[i, 16] = 1
    path_segment_incidence[i, 17] = 1
    path_segment_incidence[i, 18] = 1
    i += 1

    # 22 OD>14-1 1: L3>14-13 L2>13-10-6-5-1 (lla, bb, bm, nm)
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    boarding_flag[i, 4] = 1
    path_segment_incidence[i, 4] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    path_segment_incidence[i, 7] = 1
    i += 1
    # 23 OD>14-1 2: L4>14-11 L7>11-10-6-5-1 (bm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[19, 26] = 1
    boarding_flag[i, 42] = 1
    path_segment_incidence[i, 42] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    path_segment_incidence[i, 45] = 1
    i += 1
    # 24 OD>14-1 3: L4>14-11 L6>11-10 L8>10-9 L5>9-5-1 (bb, bm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    boarding_flag[i, 37] = 1
    path_segment_incidence[i, 37] = 1
    boarding_flag[i, 47] = 1
    path_segment_incidence[i, 47] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    path_segment_incidence[i, 35] = 1
    i += 1
    # 25 OD>14-1 4: L3>14-13-12-9-8-4-1 (lla, bb, bm, nm)
    boarding_flag[i, 14] = 1
    path_segment_incidence[i, 14] = 1
    path_segment_incidence[i, 15] = 1
    path_segment_incidence[i, 16] = 1
    path_segment_incidence[i, 17] = 1
    path_segment_incidence[i, 18] = 1
    path_segment_incidence[i, 19] = 1
    i += 1
    # 26 OD>14-1 5: L4>14-11-7-6-3-2-1 (lla, bb, bm, nm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    path_segment_incidence[i, 27] = 1
    path_segment_incidence[i, 28] = 1
    path_segment_incidence[i, 29] = 1
    path_segment_incidence[i, 30] = 1
    path_segment_incidence[i, 31] = 1
    i += 1
    # 27 OD>14-1 6: L4>14-11 L1>11-10-9-5-1 (nm)
    boarding_flag[i, 26] = 1
    path_segment_incidence[i, 26] = 1
    boarding_flag[i, 52] = 1
    path_segment_incidence[i, 52] = 1
    path_segment_incidence[i, 53] = 1
    path_segment_incidence[i, 54] = 1
    path_segment_incidence[i, 55] = 1
    i += 1

    # 28 OD>10-5 1: L2>10-6-5 (lla, bb, bm, nm)
    boarding_flag[i, 5] = 1
    path_segment_incidence[i, 5] = 1
    path_segment_incidence[i, 6] = 1
    i += 1
    # 29 OD>10-5 2: L8>10-9 L5>9-5 (bb, bm)
    boarding_flag[i, 47] = 1
    path_segment_incidence[i, 47] = 1
    boarding_flag[i, 34] = 1
    path_segment_incidence[i, 34] = 1
    i += 1
    # 30 OD>10-5 3: L7>10-6-5 (bm)
    boarding_flag[i, 43] = 1
    path_segment_incidence[i, 43] = 1
    path_segment_incidence[i, 44] = 1
    i += 1
    # 31 OD>10-5 4: L1>10-9-5 (nm)
    boarding_flag[i, 53] = 1
    path_segment_incidence[i, 53] = 1
    path_segment_incidence[i, 54] = 1
    i += 1

    # number of paths that each OD has
    od_num_paths = [4, 5, 2, 5, 4, 2, 6, 4]
    od_path2index = [[0,1,2,3], # one added at end
                     [4,5,6,7,8], # one added at end
                     [9,10],
                     [11,12,13,14,15], # one added at end
                     [16,17,18,19], # one added at end
                     [20,21],
                     [22,23,24,25,26,27], # one added at end
                     [28,29,30,31]] # one added at end
    #path2od = []
    #path2odindex = []
    #for od in range(8):
    #    for path in range(od_num_paths[od]):
    #        path2od.append(od)
    #        path2odindex.append(path)

    path_avail_lla = [[1,0,0,0],
                      [1,1,1,0,0],
                      [1,1],
                      [0,0,1,0,0],
                      [0,1,1,0],
                      [1,1],
                      [1,0,0,1,1,0],
                      [1,0,0,0]]

    path_avail_bb = [[1,1,0,0],
                     [1,1,1,0,0],
                     [1,1],
                     [1,0,1,0,0],
                     [0,1,1,0],
                     [1,1],
                     [1,0,1,1,1,0],
                     [1,1,0,0]]

    path_avail_bm = [[1,1,1,0],
                     [1,1,1,1,0],
                     [1,1],
                     [1,1,1,1,0],
                     [1,1,1,0],
                     [1,1],
                     [1,1,1,1,1,0],
                     [1,1,1,0]]

    path_avail_nm = [[1,0,0,1],
                     [1,0,1,0,1],
                     [1,1],
                     [0,0,1,0,1],
                     [0,1,0,1],
                     [1,1],
                     [1,0,0,1,1,1],
                     [1,0,0,1]]
    return num_paths, path_segment_incidence, boarding_flag, od_num_paths, od_path2index, path_avail_lla, path_avail_bb, path_avail_bm, path_avail_nm

