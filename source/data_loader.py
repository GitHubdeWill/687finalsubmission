import random
import numpy as np
import csv

def import_data(filepath):
    STATE = 0
    ACTION = 1
    REWARD = 2

    state_dimension = -1
    num_actions = -1
    iOrder = -1
    theta_b = None
    data_size = -1
    pi_b = -1

    sar = {STATE:[], ACTION:[], REWARD:[]}  # Stores state action reward in three lists

    with open(filepath) as f:
        items = [line for line in f]
        for i, row in enumerate(items): 
            if i == 0:
                state_dimension = int(row)
            elif i == 1:
                num_actions = int(row)
            elif i == 2:
                iOrder = int(row)
            elif i == 3:
                theta_b = np.fromstring(row, dtype=float, sep=',')
            elif i == 4:
                data_size = int(row)
            elif i >= 5 and i <= 5 + data_size - 1:
                temp = np.fromstring(row, dtype=float, sep=',')
                for j in sar.keys():
                    sar[j].append([])
                for index in range(len(temp)):
                    sar[index % 3][-1].append(temp[index])
            else:
                pi_b = np.array(np.fromstring(row, dtype=float, sep=','))
    # print(sar)
    return state_dimension, num_actions, iOrder, theta_b, data_size, sar[STATE], sar[ACTION], sar[REWARD], pi_b
