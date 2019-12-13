import numpy as np

from mdp.agents import skeleton as base_agent
from mdp.environments import skeleton as base_env
from mdp.environments import gridworld as gw
from mdp.environments import cartpole as cp
from mdp.policies import skeleton as base_policy
from mdp.policies import tabular_softmax as tabp
from mdp.policies import cont_policy as cont

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s; %(levelname)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import inspect
import re
def slog(x):
    """log var status with name"""
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} = {}".format(r,x))

class tabp_gridw_evaluation:
    def __init__(self, p:cont.PolySoftmax, env:cp.Cartpole):
        self.Pol = p
        self.Env = env
        self.results = []

    def __call__(self, policy_param, num_episodes):
        # logger.info("Initializing Policy and World...")
        world = self.Env()
        policy = self.Pol(25, 4)

        # print(policy_param)
        # logger.info("Setting policy Params...")
        policy.parameters=policy_param
        # print(policy.parameters)
        # print(policy._theta)
        # print(policy.getActionProbabilities(1))

        # slog(self.Pol.parameters)

        J_hats = []
        steps = []

        # logger.info("evaluating for "+str(num_episodes)+" episodes")

        for n in range(num_episodes):
            world.reset()
            ret = 0
            step = 0
            while not world.isEnd:
                step += 1
                action = policy.sampleAction(world.state)
                world.step(action)
                # logger.info("Taking "+str(action)+" ends up in state "+str(world.state))
                discounted_reward = world.reward * (world.gamma**step)
                # logger.info("reward: "+str(discounted_reward))
                ret += discounted_reward

            # logger.info("Episode "+str(n)+" Terminated w/ return "+str(ret))

            J_hats.append(ret)
            self.results.append(ret)
            steps.append(step)
            world.reset()
        # logger.debug("Avg Step: "+ str(np.mean(steps)))
        # self.results.append(np.mean(J_hats))
        return np.mean(J_hats)


class tabp_carp_evaluation:
    def __init__(self, p:tabp.TabularSoftmax, env:cp.Cartpole):
        self.Pol = p
        self.Env = env
        self.results = []

    def __call__(self, policy_param, num_episodes):
        # logger.info("Initializing Policy and World...")
        world = self.Env()
        policy = self.Pol()

        # logger.info("Setting policy Params...")
        policy.parameters=policy_param

        J_hats = []
        steps = []

        # logger.info("evaluating for "+str(num_episodes)+" episodes")

        for n in range(num_episodes):
            ret = 0
            step = 0
            while not world.isEnd:
                step += 1
                action = policy.sampleAction(world.state)
                world.step(action)
                # logger.info("Taking "+str(action)+" ends up in state "+str(world.state))
                discounted_reward = world.reward #* world.gamma**step
                # logger.info("reward: "+str(discounted_reward))
                ret += discounted_reward

            # logger.info("Episode "+str(n)+" Terminated w/ return "+str(ret))

            J_hats.append(ret)
            self.results.append(ret)
            steps.append(step)
            world.reset()
        # logger.debug("Avg Step: "+ str(np.mean(steps)))
        return np.mean(J_hats)