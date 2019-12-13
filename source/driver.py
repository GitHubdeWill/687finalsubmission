import inspect
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np

from data_loader import import_data
from hcope import HCOPE
from mdp.environments import cartpole as cp
from mdp.environments import gridworld_simple as gw
from mdp.history import History
from mdp.policies.f_policy import FBSoftmax
from policy_evaluation import PolicyEvaluation
import sys

# np.random.seed(1213971)


def slog(x):
    """log var status with name"""
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} = {}".format(r,x))


MODE = int(sys.argv[1])  # 0 for self generated; 1 for submission

actual_data_size = 20000
outputname = str(actual_data_size)+"-"+str(time.time())
if MODE == 1:

    np.random.seed(int(time.time()))
    # Load data from CSV
    state_dimension, num_actions, iOrder, theta_b, data_size, states, actions, rewards, pi_b = import_data("../data/test_data/data.csv")
    print("Loaded data.")
    print(state_dimension, num_actions, iOrder, theta_b, data_size)
    D = []

    # Parameter to select
    c = 2
    dOrder = 0
    iteration_to_run = 1000
    
    # Create Data set
    for i in range(data_size):
        history = History(states[i], actions[i], rewards[i])
        D.append(history)
    
    # Shuffle the data
    np.random.shuffle(D)

    D = D[:actual_data_size]
    slog(len(D))

    J_pi_b_hat = 0.0
    for H in D:
        J_pi_b_hat += np.sum(H.rewards)
    J_pi_b_hat /= len(D)
    slog(J_pi_b_hat)
    
    policy = FBSoftmax(state_dimension,num_actions,iOrder,dOrder)
    policy.parameters = theta_b
    safety_data_size = int(len(D)*0.6)

    inittheta = np.random.rand(4) * 10 - 5
    slog(inittheta)
    hc = HCOPE(D, policy, safety_data_size, 2, init_theta=inittheta)

    counter = 0
    # Keep sample candidate c
    for i in range(iteration_to_run):
        theta_res_pairs = hc.sample_theta_c()

        for theta, function_value, pdis_on_safety_data in theta_res_pairs:
            print("#####################",counter)
            slog(theta)
            slog(function_value)
            slog(pdis_on_safety_data)
            print("#####################",counter)
            l = [theta, function_value, pdis_on_safety_data]
            with open("./output/"+outputname+".output", "a+") as f:
                f.write(str(l))
                f.write("\n")
            counter+=1
    
elif MODE == 0:
    
    state_dimension = 1
    num_actions = 4
    iOrder = 1
    dOrder = 0

    J_pi_b_hat = -100
    while (J_pi_b_hat < -20):
        time.sleep(1)
        np.random.seed(int(time.time()))
        theta_b = (np.random.rand(8)-np.random.rand()) * (np.power(10,np.random.rand()*2))
        theta_b = np.array(theta_b)
        slog(theta_b)
        policy = FBSoftmax(state_dimension,num_actions,iOrder,dOrder)
        policy.parameters = theta_b
        history_size = 1000
        evaluation_iter = 1000
        iteration_to_run = 100
        environment = gw.Gridworld_Simple
        evaluator = PolicyEvaluation(policy, environment)
        slog(policy.parameters.shape)
        # Get Data by running the policy
        J_pi_b_hat, D, maxJ = evaluator.run_policy(theta_b, history_size)
        slog(J_pi_b_hat)
        slog(maxJ)

    c = J_pi_b_hat
    safety_data_size = int(len(D)*0.6)
     
    hc = HCOPE(D, policy, safety_data_size, c)
    counter = 0
    # Keep sample candidate c
    for i in range(iteration_to_run):
        theta_res_pairs = hc.sample_theta_c()

        for theta, function_value, pdis_on_safety_data in theta_res_pairs:
            print("#####################",counter)
            slog(theta)
            slog(function_value)
            slog(pdis_on_safety_data)

            # Actual J
            test_policy = FBSoftmax(state_dimension,num_actions,iOrder,dOrder)
            test_policy.parameters = theta
            test_pdis = hc.test_pdis(test_policy)
            evaluator = PolicyEvaluation(test_policy, environment)

            mean_J_hat_after, _, _ = evaluator.run_policy(theta.reshape(test_policy.parameters.shape), evaluation_iter)
            slog(mean_J_hat_after)
            print("#####################",counter)
            l = [theta, function_value, pdis_on_safety_data, mean_J_hat_after]
            with open("./output/gridworld"+str(outputname)+".output", "a+") as f:
                f.write(str(l))
                f.write("\n")
            time.sleep(1)
            counter+=1
        
