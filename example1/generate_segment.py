import numpy as np


def segment_generation():
    '''generate segments'''
    # return num_lines, segment_cost, segment_line
    # num_lines: integer
    # segment_cost[s]: cost of segment s
    # segment_line[s]: transit line of segment s

    # list of transit segments:
    # (sequencing: L2_dir0, L2_dir1, L3_dir0, L3_dir1,...)
    #        0   1   2     3
    # (L2): 1-5 5-6 6-10 10-13
    #        4      5    6   7
    # (L2): 13-10  10-6 6-5 5-1
    #        8   9  10   11   12    13
    # (L3): 1-4 4-8 8-9 9-12 12-13 13-14
    #        14    15    16  17  18  19
    # (L3): 14-13 13-12 12-9 9-8 8-4 4-1
    #        20  21  22  23  24   25
    # (L4): 1-2 2-3 3-6 6-7 7-11 11-14
    #        26    27   28  29  30  31
    # (L4): 14-11 11-7 7-6 6-3 3-2 2-1
    #       32  33
    # (L5): 1-5 5-9
    #       34  35
    # (L5): 9-5 5-1
    #        36
    # (L6): 10-11
    #        37
    # (L6): 11-10
    #       38  39   40   41
    # (L7): 1-5 5-6 6-10 10-11
    #        42    43  44  45
    # (L7): 11-10 10-6 6-5 5-1
    #        46
    # (L8): 9-10
    #        47
    # (L8): 10-9
    num_segments = 48
    segment_cost = np.array([6, 4, 4, 4,
                             4, 4, 4, 6,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             6, 4,
                             4, 6,
                             4,
                             4,
                             6, 4, 4, 4,
                             4, 4, 4, 6,
                             8,
                             8])
    # Note: Line L1's segment is not included yet

    segment_line = [1, 1, 1, 1,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3,
                    4, 4,
                    4, 4,
                    5,
                    5,
                    6, 6, 6, 6,
                    6, 6, 6, 6,
                    7,
                    7]

    return num_segments, segment_cost, segment_line


def segment_generation_expanded():
    '''generate segments with disrupted line segments included'''
    # return num_lines, segment_cost, segment_line
    # num_lines: integer
    # segment_cost[s]: cost of segment s
    # segment_line[s]: transit line of segment s

    # list of transit segments:
    # (sequencing: L2_dir0, L2_dir1, L3_dir0, L3_dir1,...
    #  L0 is appended to the end)
    #        0   1   2     3
    # (L2): 1-5 5-6 6-10 10-13
    #        4      5    6   7
    # (L2): 13-10  10-6 6-5 5-1
    #        8   9  10   11   12    13
    # (L3): 1-4 4-8 8-9 9-12 12-13 13-14
    #        14    15    16  17  18  19
    # (L3): 14-13 13-12 12-9 9-8 8-4 4-1
    #        20  21  22  23  24   25
    # (L4): 1-2 2-3 3-6 6-7 7-11 11-14
    #        26    27   28  29  30  31
    # (L4): 14-11 11-7 7-6 6-3 3-2 2-1
    #       32  33
    # (L5): 1-5 5-9
    #       34  35
    # (L5): 9-5 5-1
    #        36
    # (L6): 10-11
    #        37
    # (L6): 11-10
    #       38  39   40   41
    # (L7): 1-5 5-6 6-10 10-11
    #        42    43  44  45
    # (L7): 11-10 10-6 6-5 5-1
    #        46
    # (L8): 9-10
    #        47
    # (L8): 10-9
    # (L1): 1-5 5-9 9-10 10-11
    #       48  49   50   51
    # (L1): 11-10 10-9 9-5 5-1
    #       52     52  54   55
    num_segments = 56
    segment_cost = np.array([6, 4, 4, 4,
                             4, 4, 4, 6,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             8, 8, 8, 8, 8, 8,
                             6, 4,
                             4, 6,
                             4,
                             4,
                             6, 4, 4, 4,
                             4, 4, 4, 6,
                             8,
                             8,
                             6, 4, 4, 4,
                             4, 4, 4, 6])
    # Note: Line L1's segment is appended to the end
    # L1 is available only after disruption recovers

    segment_line = [1, 1, 1, 1,
                    1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                    3, 3, 3, 3, 3, 3,
                    4, 4,
                    4, 4,
                    5,
                    5,
                    6, 6, 6, 6,
                    6, 6, 6, 6,
                    7,
                    7,
                    0, 0, 0, 0,
                    0, 0, 0, 0]
    # Note: line L_i has index (i-1)

    return num_segments, segment_cost, segment_line
